#!/usr/bin/env python3
"""Real cone detector using the supplied duba.pt model; publishes /perception/cones."""
import json, os, cv2, rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO
class ConeDetector(Node):
    def __init__(self):
        super().__init__("cone_detector_node")
        self.declare_parameter("model_path", os.path.join(os.path.dirname(os.path.abspath(__file__)),"duba.pt"))
        self.declare_parameter("camera_topic","/carla/ego_vehicle/rgb_front/image")
        self.declare_parameter("confidence_threshold",0.70)
        self.model=YOLO(str(self.get_parameter("model_path").value)); self.threshold=float(self.get_parameter("confidence_threshold").value); self.bridge=CvBridge()
        self.pub=self.create_publisher(String,"/perception/cones",10); self.debug=self.create_publisher(Image,"/perception/cone_debug_hud",10)
        self.create_subscription(Image,str(self.get_parameter("camera_topic").value),self.cb,qos_profile_sensor_data)
        self.get_logger().info("Duba algılama başladı.")
    def cb(self,msg):
        try:
            frame=self.bridge.imgmsg_to_cv2(msg,"bgr8"); results=self.model(frame,verbose=False,device="cpu"); annotated=frame.copy(); detections=[]
            for r in results:
                for box in r.boxes:
                    conf=float(box.conf[0])
                    if conf<self.threshold: continue
                    x1,y1,x2,y2=[int(v) for v in box.xyxy[0].tolist()]; cls=str(self.model.names[int(box.cls[0])])
                    detections.append({"class":cls,"confidence":round(conf,3),"bbox":[x1,y1,x2,y2],"center_px":[(x1+x2)//2,(y1+y2)//2]})
                    cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,0,255),2); cv2.putText(annotated,"%s %.2f"%(cls,conf),(x1,max(24,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2)
            out=String(); out.data=json.dumps({"timestamp_sec":msg.header.stamp.sec,"timestamp_nanosec":msg.header.stamp.nanosec,"detections":detections}); self.pub.publish(out)
            dbg=self.bridge.cv2_to_imgmsg(annotated,"bgr8"); dbg.header=msg.header; self.debug.publish(dbg)
            cv2.imshow("Duba Algilama",cv2.resize(annotated,(854,480))); cv2.waitKey(1)
        except Exception as e:self.get_logger().error("Duba algılama hatası: %s"%e,throttle_duration_sec=1.0)
def main(args=None):
    rclpy.init(args=args); node=ConeDetector()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        cv2.destroyAllWindows(); node.destroy_node(); rclpy.shutdown()
if __name__=="__main__":main()
