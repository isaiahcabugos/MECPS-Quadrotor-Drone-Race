#!/bin/bash
set -e

# ================================
# CONFIGURATION
# ================================
NUM_RUNS=200
SAVE_DIR="$HOME/mpc_data"
DATA_PROCESS_SCRIPT="$HOME/combine_csv.py"
MODEL_TRAIN_SCRIPT="$HOME/train_model.py"

mkdir -p "$SAVE_DIR"

# ================================
# DATA COLLECTION LOOP
# ================================
echo "===================================="
echo "Starting data collection in Gazebo"
echo "===================================="

for ((i=1; i<=NUM_RUNS; i++)); do
  echo "========== RUN $i / $NUM_RUNS =========="

  RUN_TAG="RUN_$i"

  # Launch Gazebo
  gnome-terminal -- bash -c "echo '[$RUN_TAG] Gazebo starting...'; roslaunch clover_simulation simulator.launch gui:=true" &
  sleep 40

  # Launch Code to hover
  gnome-terminal -- bash -c "echo '[$RUN_TAG] Hover code starting...'; rosrun clover_3d_mpc takeoff_hover.py" &
  sleep 13

  # Launch MPC
  gnome-terminal -- bash -c "echo '[$RUN_TAG] MPC starting...'; rosrun clover_3d_mpc mpc_3d_node.py" &
  sleep 5

  echo "Run $i in progress..."
  sleep 40

  # --- Kill everything after run ---
  echo "Closing processes for $RUN_TAG..."
  pkill -f gzserver || true
  pkill -f gzclient || true
  pkill -f roslaunch || true
  pkill -f rosrun || true

  echo "Run $i complete. Cooling down..."
  sleep 10
done

echo "All $NUM_RUNS simulation runs complete!"
echo "Data saved in: $SAVE_DIR"

# ================================
# COMBINE AND PROCESS CSV FILES
# ================================
echo
echo "===================================="
echo "Combining CSV files and preprocessing data"
echo "===================================="

if [ -f "$DATA_PROCESS_SCRIPT" ]; then
  python3 "$DATA_PROCESS_SCRIPT"
else
  echo "Error: Could not find $DATA_PROCESS_SCRIPT"
  exit 1
fi

echo "Data preprocessing complete."

# ================================
# TRAIN MODEL
# ================================
echo
echo "===================================="
echo "Starting model training"
echo "===================================="

if [ -f "$MODEL_TRAIN_SCRIPT" ]; then
  python3 "$MODEL_TRAIN_SCRIPT"
else
  echo "Error: Could not find $MODEL_TRAIN_SCRIPT"
  exit 1
fi

echo
echo "===================================="
echo "   Full pipeline complete!"
echo "   - Simulation data collected"
echo "   - CSVs combined and processed"
echo "   - Model trained and saved"
echo "===================================="