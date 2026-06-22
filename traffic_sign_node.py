import os
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import json
from ultralytics import YOLO

class TrafficSignDetectorNode(Node):
    def __init__(self):
        super().__init__('traffic_sign_detector_node')
        
        # 1. ROS 2 Parametreleri
        _default_model = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'denemetabela.pt')
        self.declare_parameter('model_path', _default_model)
        self.declare_parameter('camera_topic', '/carla/ego_vehicle/rgb_front/image')
        
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        
        # 2. YOLO Modelini Yükle
        self.get_logger().info(f'YOLO Modeli Yukleniyor: {model_path}')
        try:
            self.model = YOLO(model_path)
            self.get_logger().info('Model basariyla yuklendi.')
        except Exception as e:
            self.get_logger().error(f'Model yukleme hatasi: {e}')
            
        self.bridge = CvBridge()
        
        # 3. Abonelikler ve Yayıncılar
        self.subscription = self.create_subscription(
            Image, 
            camera_topic, 
            self.image_callback, 
            qos_profile_sensor_data
        )
        
        self.sign_publisher = self.create_publisher(String, '/perception/traffic_signs', 10)
        self.debug_image_publisher = self.create_publisher(Image, '/perception/sign_debug_hud', 10)

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # YOLO Çıkarımı (Önceki adımdaki hatayı önlemek için CPU modunda çalıştırıyoruz)
            results = self.model(frame, verbose=False, device='cpu')
            
            detections = []
            annotated_frame = frame.copy()
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]
                    
                    if conf > 0.60:
                        detections.append({
                            "class": cls_name,
                            "confidence": round(conf, 2),
                            "bbox": [int(x1), int(y1), int(x2), int(y2)]
                        })
                        
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                        label = f"{cls_name} %{int(conf*100)}"
                        cv2.putText(annotated_frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # ROS 2 Üzerinden Yayınlama İşlemleri
            sign_msg = String()
            sign_msg.data = json.dumps({
                "timestamp_sec": msg.header.stamp.sec,
                "timestamp_nanosec": msg.header.stamp.nanosec,
                "detections": detections
            })
            self.sign_publisher.publish(sign_msg)
            
            debug_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
            debug_msg.header = msg.header
            self.debug_image_publisher.publish(debug_msg)

            # =================================================================
            # YENİ EKLENEN KISIM: Görüntüyü Bağımsız Bir Pencerede Aç
            # =================================================================
            # Orijinal görüntü (720p) ekranı kaplamasın diye 480p'ye boyutlandırıyoruz
            display_frame = cv2.resize(annotated_frame, (854, 480))
            cv2.imshow("YOLO Trafik Levhasi Algilama (Canli Yayin)", display_frame)
            
            # OpenCV penceresinin donmaması için 1 milisaniyelik bekleme süresi şarttır
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Goruntu isleme hatasi: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = TrafficSignDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # ROS düğümü kapatıldığında açık kalan OpenCV pencerelerini temizle
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
