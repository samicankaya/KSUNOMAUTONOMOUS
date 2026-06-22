import os
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
import cv2
import json
from ultralytics import YOLO

class PedestrianDetectorNode(Node):
    def __init__(self):
        super().__init__('pedestrian_detector_node')
        
        # 1. ROS 2 Parametreleri
        _default_model = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best.pt')
        self.declare_parameter('model_path', _default_model)
        self.declare_parameter('camera_topic', '/carla/ego_vehicle/rgb_front/image')
        
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        
        # 2. YOLO Modelini Yükle
        self.get_logger().info(f'YOLO Yaya Modeli Yukleniyor: {model_path}')
        try:
            self.model = YOLO(model_path)
            self.get_logger().info('Yaya Modeli basariyla yuklendi.')
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
        
        self.pedestrian_publisher = self.create_publisher(String, '/perception/pedestrians', 10)
        self.debug_image_publisher = self.create_publisher(Image, '/perception/pedestrian_debug_hud', 10)
        
        # YENİ: Acil fren sinyali yayınlayıcısı
        self.emergency_brake_publisher = self.create_publisher(Bool, '/perception/emergency_brake', 10)

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # Ekran boyutlarını al
            height, width, _ = frame.shape
            
            # Sütun sınırlarını hesapla (3 eşit parça)
            col_width = width // 3
            left_col_end = col_width
            mid_col_end = col_width * 2
            
            results = self.model(frame, verbose=False, device='cpu')
            
            detections = []
            annotated_frame = frame.copy()
            
            # Acil fren bayrağı (her frame için baştan sıfırlanır)
            emergency_brake_required = False
            
            # Görüntüye sütun çizgilerini çiz (görselleştirme için opsiyonel)
            cv2.line(annotated_frame, (left_col_end, 0), (left_col_end, height), (0, 255, 255), 2)
            cv2.line(annotated_frame, (mid_col_end, 0), (mid_col_end, height), (0, 255, 255), 2)
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]
                    
                    if conf > 0.60:
                        # Bounding box'ın merkez x koordinatını hesapla
                        center_x = int((x1 + x2) / 2)
                        
                        # Bölge Kontrolü
                        position_text = ""
                        if center_x < left_col_end:
                            position_text = "Solda - Devam"
                            color = (0, 255, 0) # Yeşil
                        elif left_col_end <= center_x <= mid_col_end:
                            position_text = "Ortada- yavaslama"
                            emergency_brake_required = True
                            color = (0, 0, 255) # Kırmızı
                        else:
                            position_text = "Sagda - Devam"
                            color = (0, 255, 0) # Yeşil

                        detections.append({
                            "class": cls_name,
                            "confidence": round(conf, 2),
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "position": position_text
                        })
                        
                        # Bounding box ve merkez noktasını çiz
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                        cv2.circle(annotated_frame, (center_x, int((y1+y2)/2)), 5, color, -1)
                        
                        # Etiket yazdır
                        label = f"{cls_name} {int(conf*100)}% - {position_text}"
                        cv2.putText(annotated_frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 1. ROS 2'ye JSON olarak algılamaları bas
            pedestrian_msg = String()
            pedestrian_msg.data = json.dumps({
                "timestamp_sec": msg.header.stamp.sec,
                "timestamp_nanosec": msg.header.stamp.nanosec,
                "detections": detections
            })
            self.pedestrian_publisher.publish(pedestrian_msg)
            
            # 2. ROS 2'ye Acil Fren sinyalini bas
            brake_msg = Bool()
            brake_msg.data = emergency_brake_required
            self.emergency_brake_publisher.publish(brake_msg)
            
            # Konsola da bilgi verelim
            if emergency_brake_required:
                self.get_logger().warn('DIKKAT: Yaya orta seritte. ACIL FREN komutu gonderildi!', throttle_duration_sec=1.0)
            
            # 3. Hata ayıklama (Debug) görüntüsünü ROS 2'ye bas
            debug_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
            debug_msg.header = msg.header
            self.debug_image_publisher.publish(debug_msg)

            # 4. Görüntüyü OpenCV penceresinde göster
            display_frame = cv2.resize(annotated_frame, (854, 480))
            cv2.imshow("Insan Algilama Penceresi", display_frame) 
            
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Goruntu isleme hatasi: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = PedestrianDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
