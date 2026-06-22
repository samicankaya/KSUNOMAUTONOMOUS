import cv2
import numpy as np
import math
import csv
import json
from scipy.signal import find_peaks

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

# =========================================================================
# ÇOKLU ŞERİT VE ÇEVRESEL FARKINDALIK ALGI KATMANI (SOBEL-X ENTEGRELİ)
# =========================================================================

class MultiLanePerception:
    def __init__(self, debug_mode=True, img_w=1280, img_h=720, fov=110.0, cam_h=1.5, pitch=0.0):
        self.debug_mode = debug_mode
        self.img_w = img_w
        self.img_h = img_h
        
        # 1. LENS DİSTORSİYON KALİBRASYONU
        self.mtx = np.array([[900.0, 0.0, img_w/2], [0.0, 900.0, img_h/2], [0.0, 0.0, 1.0]])
        self.dist = np.array([[-0.24, 0.05, -0.001, 0.002, -0.005]])
        
        # 2. DİNAMİK IPM (KUŞ BAKIŞI) HESAPLAMASI
        self.calculate_ipm_matrix(fov_deg=fov, img_w=img_w, img_h=img_h, cam_height_m=cam_h, pitch_deg=pitch)
        
        # 3. EGO ŞERİT İÇİN KALMAN FİLTRESİ
        self.kf = cv2.KalmanFilter(6, 6)
        self.kf.transitionMatrix = np.eye(6, dtype=np.float32)
        self.kf.measurementMatrix = np.eye(6, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-4
        self.kf.measurementNoiseCov = np.eye(6, dtype=np.float32) * 1e-1
        self.is_kf_initialized = False
        
        self.ploty = np.linspace(0, self.img_h - 1, self.img_h) 
        self.frame_count = 0

    def calculate_ipm_matrix(self, fov_deg, img_w, img_h, cam_height_m, pitch_deg):
        """Kamera parametrelerine göre Kuş Bakışı (IPM) matrisini hesaplar."""
        pitch_rad = math.radians(pitch_deg)
        fov_rad = math.radians(fov_deg)
        
        f = (img_w / 2.0) / math.tan(fov_rad / 2.0)
        
        d_min = 3.5   
        d_max = 25.0  
        lat_w = 6.0   
        
        world_pts = [(-lat_w, d_min), (lat_w, d_min), (lat_w, d_max), (-lat_w, d_max)]
        src_points = []
        
        for X_world, Y_world in world_pts:
            Y_cam = cam_height_m * math.cos(pitch_rad) - Y_world * math.sin(pitch_rad)
            Z_cam = cam_height_m * math.sin(pitch_rad) + Y_world * math.cos(pitch_rad)
            
            u = f * (X_world / Z_cam) + (img_w / 2.0)
            v = f * (Y_cam / Z_cam) + (img_h / 2.0)
            src_points.append([u, v])
            
        self.src_pts = np.float32(src_points)
        
        self.dst_pts = np.float32([
            [img_w * 0.25, img_h],       
            [img_w * 0.75, img_h],       
            [img_w * 0.75, 0],           
            [img_w * 0.25, 0]            
        ])
        
        self.M = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)
        self.Minv = cv2.getPerspectiveTransform(self.dst_pts, self.src_pts)
        
        self.xm_per_pix = (lat_w * 2) / (img_w * 0.5)
        self.ym_per_pix = (d_max - d_min) / img_h

    def undistort_image(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    def preprocess_bev(self, bev_color):
        """Gölgeleri yenmek için Renk ve Sobel-X (Türev) Gradyanını birleştiren algoritma"""
        hls = cv2.cvtColor(bev_color, cv2.COLOR_BGR2HLS)
        
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]
        
        # 1. SOBEL-X GRADYANI: Ani parlaklık değişimi olan dikey kenarları bulur (Gölgeleri yok sayar)
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=5)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel_x = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        
        sxbinary = np.zeros_like(scaled_sobel_x)
        sxbinary[(scaled_sobel_x >= 30) & (scaled_sobel_x <= 255)] = 255
        
        sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=5)
        abs_sobely = np.absolute(sobely)
        scaled_sobel_y = np.uint8(255 * abs_sobely / np.max(abs_sobely))
        
        sybinary = np.zeros_like(scaled_sobel_y)
        sybinary[(scaled_sobel_y >= 30) & (scaled_sobel_y <= 255)] = 255
        
        # 2. RENK EŞİKLEME: Sarı ve beyaz renkleri yakalar
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= 170) & (s_channel <= 255)] = 255
        
        l_binary = np.zeros_like(l_channel)
        l_binary[(l_channel >= 210) & (l_channel <= 255)] = 255
        
        # 3. BİRLEŞTİRME: Gradyan (kenar) veya Renk maskesinden herhangi biri şeridi tespit ederse kabul et
        combined = np.zeros_like(sxbinary)
        combined[(s_binary == 255) | (l_binary == 255) | (sxbinary == 255) | (sybinary == 255)] = 255
        
        # 4. TEMİZLİK VE BAĞLANTI 
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_clean)
        
        return cleaned

    def find_all_lines(self, binary_bev):
        out_img = np.dstack((binary_bev, binary_bev, binary_bev)) * 255
        
        # MASKELEME
        mask_height = int(binary_bev.shape[0] * 0.40)
        binary_bev[0:mask_height, :] = 0  # Ufuk çizgisi
        binary_bev[-60:, :] = 0           # Kaput hizası
        
        nonzero = binary_bev.nonzero()
        nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
        margin = 120
        minpix = 15    
        maxpix = 400   

        # ZİRVE BULUCU (Histogram)
        histogram = np.sum(binary_bev[int(binary_bev.shape[0]//4):, :], axis=0)
        peaks, _ = find_peaks(histogram, height=15, distance=120)
        
        detected_lines = []
        
        for peak_x in peaks:
            current_x = peak_x
            line_inds = []
            
            for window in range(9):
                win_y_low = binary_bev.shape[0] - (window + 1) * int(binary_bev.shape[0] // 9)
                win_y_high = binary_bev.shape[0] - window * int(binary_bev.shape[0] // 9)
                
                good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                             (nonzerox >= current_x - margin) & (nonzerox < current_x + margin)).nonzero()[0]
                
                line_inds.append(good_inds)
                
                # Çok büyük bloklar kayan pencereyi saptırmasın
                if minpix < len(good_inds) < maxpix:
                    current_x = int(np.mean(nonzerox[good_inds]))
            
            line_inds = np.concatenate(line_inds)
            
            if len(line_inds) > 150: 
                line_x = nonzerox[line_inds]
                line_y = nonzeroy[line_inds]
                try:
                    fit = np.polyfit(line_y, line_x, 2)
                    
                    x_bottom = fit[0]*(self.img_h**2) + fit[1]*self.img_h + fit[2]
                    x_top = fit[0]*(0**2) + fit[1]*0 + fit[2]
                    
                    is_curve_sane = abs(fit[0]) < 0.005
                    
                    if abs(x_bottom - x_top) < 400 and is_curve_sane:  
                        detected_lines.append(fit)
                        out_img[line_y, line_x] = [0, 255, 255]
                except:
                    pass
                    
        return detected_lines, out_img

    def cluster_lanes(self, lines, img_h, img_w):
        if not lines:
            return None, None, [], []
            
        bottom_x_vals = []
        for fit in lines:
            x_val = fit[0]*(img_h**2) + fit[1]*img_h + fit[2]
            bottom_x_vals.append((x_val, fit))
            
        bottom_x_vals.sort(key=lambda item: item[0])
        
        car_center = img_w / 2
        ego_left, ego_right = None, None
        adjacent_left, adjacent_right = [], []
        
        left_lines = [item for item in bottom_x_vals if item[0] < car_center]
        right_lines = [item for item in bottom_x_vals if item[0] >= car_center]
        
        if left_lines:
            ego_left = left_lines[-1][1] 
            adjacent_left = [item[1] for item in left_lines[:-1]] 
            
        if right_lines:
            ego_right = right_lines[0][1] 
            adjacent_right = [item[1] for item in right_lines[1:]] 
            
        return ego_left, ego_right, adjacent_left, adjacent_right

    def process_perception(self, frame):
        self.frame_count += 1
        img_h, img_w = frame.shape[:2]
        
        if img_w != self.img_w or img_h != self.img_h:
            self.img_w = img_w
            self.img_h = img_h
            self.ploty = np.linspace(0, self.img_h - 1, self.img_h)
            self.calculate_ipm_matrix(fov_deg=110.0, img_w=img_w, img_h=img_h, cam_height_m=1.5, pitch_deg=0.0)
        
        undist_img = self.undistort_image(frame)
        bev_color = cv2.warpPerspective(undist_img, self.M, (img_w, img_h), flags=cv2.INTER_LINEAR)
        binary_bev = self.preprocess_bev(bev_color)
        
        all_lines, sliding_img = self.find_all_lines(binary_bev)
        e_left, e_right, adj_left, adj_right = self.cluster_lanes(all_lines, img_h, img_w)
        
        ego_state = "LANE_LOST"
        lat_err = 0.0
        heading_err = 0.0
        
        if e_left is not None and e_right is not None:
            ego_state = "TRACKING_OK"
            measurement = np.array([[e_left[0]], [e_left[1]], [e_left[2]], [e_right[0]], [e_right[1]], [e_right[2]]], dtype=np.float32)
            if not self.is_kf_initialized:
                self.kf.statePost = measurement
                self.is_kf_initialized = True
            self.kf.predict()
            self.kf.correct(measurement)
            final_state = self.kf.statePost
            
            e_left = final_state[0:3].flatten()
            e_right = final_state[3:6].flatten()
            
            lane_center_px = (e_left[2] + e_right[2]) / 2
            lat_err = float((img_w / 2 - lane_center_px) * self.xm_per_pix)
            heading_err = float(math.atan((e_left[1] + e_right[1]) / 2.0))
        else:
            self.kf.predict()
            if self.is_kf_initialized:
                ego_state = "DEGRADED_TRACKING"
                final_state = self.kf.statePre
                e_left = final_state[0:3].flatten()
                e_right = final_state[3:6].flatten()

        has_right_lane = len(adj_right) > 0
        has_left_lane = len(adj_left) > 0

        fsm_message = {
            "timestamp_frame": self.frame_count,
            "ego_lane": {
                "state": ego_state,
                "lateral_error_m": round(lat_err, 4),
                "heading_error_rad": round(heading_err, 4)
            },
            "adjacent_lanes": {
                "left_lane_detected": has_left_lane,
                "right_lane_detected": has_right_lane,
                "total_lines_detected": len(all_lines)
            },
            "maneuver_clearance": {
                "safe_to_dodge_left": has_left_lane,
                "safe_to_dock_right": has_right_lane
            }
        }

        debug_panel = None
        if self.debug_mode:
            debug_panel = self.draw_debug_hud(undist_img, sliding_img, e_left, e_right, adj_left, adj_right, ego_state, lat_err)

        return fsm_message, debug_panel

    def draw_debug_hud(self, orig, sliding_img, e_l, e_r, a_l, a_r, state, lat_err):
        h, w = orig.shape[:2]
        color_warp = np.zeros_like(orig).astype(np.uint8)
        
        if e_l is not None and e_r is not None:
            left_fitx = e_l[0]*self.ploty**2 + e_l[1]*self.ploty + e_l[2]
            right_fitx = e_r[0]*self.ploty**2 + e_r[1]*self.ploty + e_r[2]
            pts_left = np.array([np.transpose(np.vstack([left_fitx, self.ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, self.ploty])))])
            pts = np.hstack((pts_left, pts_right))
            fill_color = (0, 255, 0) if state == "TRACKING_OK" else (0, 255, 255)
            cv2.fillPoly(color_warp, np.int_([pts]), fill_color)

        for adj_line in a_l + a_r:
            adj_fitx = adj_line[0]*self.ploty**2 + adj_line[1]*self.ploty + adj_line[2]
            pts_adj = np.array([np.transpose(np.vstack([adj_fitx, self.ploty]))], np.int32)
            cv2.polylines(color_warp, [pts_adj], isClosed=False, color=(255, 100, 0), thickness=15)

        newwarp = cv2.warpPerspective(color_warp, self.Minv, (w, h))
        main_view = cv2.addWeighted(orig, 1, newwarp, 0.4, 0)
        
        cv2.polylines(main_view, [np.int32(self.src_pts)], True, (0,0,255), 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(main_view, f"Ego State: {state}", (20, 40), font, 1, (255,255,255), 2)
        cv2.putText(main_view, f"Lat Err: {lat_err:.2f}m", (20, 80), font, 1, (255,255,255), 2)
        cv2.putText(main_view, f"Left Lane: {'Yes' if a_l else 'No'}", (20, 120), font, 1, (255,100,0), 2)
        cv2.putText(main_view, f"Right Lane/Dock: {'Yes' if a_r else 'No'}", (20, 160), font, 1, (255,100,0), 2)

        debug_panel = np.zeros((500, 1200, 3), dtype=np.uint8)
        debug_panel[50:500, 0:800] = cv2.resize(main_view, (800, 450))
        debug_panel[50:500, 800:1200] = cv2.resize(sliding_img, (400, 450))
        cv2.putText(debug_panel, "Multi-Lane Perception (Sobel-X)", (10, 30), font, 0.8, (255,255,255), 2)
        cv2.putText(debug_panel, "BEV Peaks Analysis", (810, 30), font, 0.8, (255,255,255), 2)

        return debug_panel


# =========================================================================
# ROS 2 NODE ENTEGRASYONU
# =========================================================================

class MultiLaneNode(Node):
    def __init__(self):
        super().__init__('multi_lane_perception_node')
        
        self.declare_parameter('fov', 110.0)
        self.declare_parameter('cam_height', 1.5)
        self.declare_parameter('pitch', 0.0)
        
        fov_val = self.get_parameter('fov').get_parameter_value().double_value
        cam_h_val = self.get_parameter('cam_height').get_parameter_value().double_value
        pitch_val = self.get_parameter('pitch').get_parameter_value().double_value

        self.subscription = self.create_subscription(Image, '/carla/ego_vehicle/rgb_front/image', self.image_callback, qos_profile_sensor_data)
        self.fsm_publisher = self.create_publisher(String, '/perception/lane_fsm_status', 10)
        self.debug_image_publisher = self.create_publisher(Image, '/perception/debug_hud', 10)
        
        self.bridge = CvBridge()
        
        self.tracker = MultiLanePerception(
            debug_mode=True, 
            fov=fov_val, 
            cam_h=cam_h_val, 
            pitch=pitch_val
        )
        self.get_logger().info(f'Çoklu Şerit Düğümü Başlatıldı. FOV: {fov_val}, Height: {cam_h_val}m, Pitch: {pitch_val}°')

    def image_callback(self, msg):
        self.get_logger().info('Goruntu alindi, isleniyor...', throttle_duration_sec=1.0)
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            fsm_dict, debug_hud = self.tracker.process_perception(frame)
            
            fsm_msg = String()
            fsm_msg.data = json.dumps(fsm_dict)
            self.fsm_publisher.publish(fsm_msg)

            if debug_hud is not None:
                # RViz için yayına devam et (İsteğe bağlı)
                debug_msg = self.bridge.cv2_to_imgmsg(debug_hud, encoding="bgr8")
                debug_msg.header = msg.header
                self.debug_image_publisher.publish(debug_msg)

                # DOĞRUDAN PENCEREDE GÖSTERİM KISMI
                cv2.imshow("Serit Takip Paneli", debug_hud)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Hata: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = MultiLaneNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()  # Düğüm kapandığında pencereleri temizler
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
