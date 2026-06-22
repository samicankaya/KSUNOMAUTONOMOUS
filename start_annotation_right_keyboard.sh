#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_FILE="${SETUP_FILE:-$HOME/setup_carla_ros_bridge.sh}"
INPUT_DEVICE="${1:-}"

if [[ -z "$INPUT_DEVICE" ]]; then
  echo "Kullanım:"
  echo "  $0 /dev/input/by-id/<keyboard-event-kbd>"
  echo
  echo "Cihazları görmek için:"
  echo "  ls -l /dev/input/by-id/ | grep -i event-kbd"
  exit 1
fi

if [[ ! -e "$INPUT_DEVICE" ]]; then
  echo "Hata: Klavye cihazı bulunamadı: $INPUT_DEVICE"
  exit 1
fi

source "$SETUP_FILE"
cd "$SCRIPT_DIR"
export DISPLAY="${DISPLAY:-:0}"

python3 -u operator_lane_annotation_dual_keyboard.py --input-device "$1" --ros-args \
  --params-file operator_lane_annotation_dual_keyboard.yaml \
  -p input_device:="$INPUT_DEVICE" \
  -p grab_input_device:=true
