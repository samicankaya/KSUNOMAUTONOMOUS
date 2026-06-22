#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_FILE="${SETUP_FILE:-$HOME/setup_carla_ros_bridge.sh}"
[[ -f "$SETUP_FILE" ]] || { echo "Hata: $SETUP_FILE bulunamadı."; exit 1; }
source "$SETUP_FILE"; export DISPLAY="${DISPLAY:-:0}"
open_term(){ local title="$1"; local command="$2"; gnome-terminal --title="$title" -- bash -lc "export DISPLAY='$DISPLAY'; source '$SETUP_FILE'; cd '$SCRIPT_DIR'; $command; exec bash"; }
open_term "CARLA ROS Bridge" "ros2 run carla_ros_bridge bridge --ros-args -p host:=localhost -p synchronous_mode:=False"
sleep 3
open_term "CARLA Ego ve Sensorler" "ros2 launch carla_spawn_objects carla_example_ego_vehicle.launch.py"
sleep 3
open_term "RViz2" "rviz2 -d \$HOME/carla-ros-bridge-ws/src/ros-bridge/carla_ad_demo/config/carla_ad_demo_ros2.rviz"
open_term "CARLA Manual Control" "ros2 launch carla_manual_control carla_manual_control.launch.py role_name:=ego_vehicle"
open_term "Trafik Levhasi Algilama" "python3 -u traffic_sign_node.py --ros-args -p model_path:='$SCRIPT_DIR/denemetabela.pt'"
open_term "Insan Algilama" "python3 -u humandetect.py --ros-args -p model_path:='$SCRIPT_DIR/best.pt'"
open_term "Duba Algilama" "python3 -u cone_detector_node.py --ros-args -p model_path:='$SCRIPT_DIR/duba.pt'"
open_term "Serit Takip" "python3 -u seritv11.py"
echo "CARLA otonom sürüş başlatıldı"
