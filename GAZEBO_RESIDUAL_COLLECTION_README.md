# Gazebo Residual Collection Guide

Quick guide for collecting dynamics residuals in Gazebo after completing 100M+ training steps in the lightweight simulator.

---

## Prerequisites

- ✅ Trained policy checkpoint at 100M+ steps (e.g., `policy_step_140000000`)
- ✅ Gazebo simulation with Clover package installed
- ✅ ROS workspace configured
- ✅ Python environment with TensorFlow, scikit-learn

---

## Overview

**What are residuals?**
- Your lightweight simulator: Fast velocity response (tau ≈ 0.1)
- Gazebo PX4 simulation: Realistic cascaded control (tau ≈ 1.0)
- Residuals: The difference between these two dynamics models

**Why collect residuals?**
- Bridge the gap between simplified training sim and realistic Gazebo physics
- Improve policy performance in Gazebo by fine-tuning with learned corrections
- Prepare for eventual real hardware deployment

---

## Step 1: Launch Gazebo Environment

**Terminal 1: Start Gazebo**
```bash
roslaunch clover_simulation simulator.launch
# Wait for Gazebo to fully load (~30 seconds)
```

**Terminal 2: Spawn ArUco Markers**
```bash
# For localization (simulates indoor GPS)
roslaunch clover_simulation aruco.launch
```

**Terminal 3: Spawn Gate Visualizations (Optional)**
```bash
cd /path/to/project

# Choose one visualization method:
python spawn_race_markers.py      # Sphere markers (simple)
python spawn_plane_gates.py       # Plane gates (realistic)
python spawn_frame_gates.py       # Frame gates (best)
```

**Verify Setup:**
```bash
# Check ground truth is available
rostopic echo /gazebo/model_states --noarr
# Should show: name: ['ground_plane', 'clover', ...]

# Check ArUco detection
rostopic echo /aruco_detect/markers
# Should show detected marker IDs when visible
```

---

## Step 2: Test Policy in Gazebo (Baseline)

Quick test to verify the trained policy works in Gazebo before collecting data.

```bash
cd /path/to/catkin_ws/src/clover/clover/examples

python3 model_controller_with_residuals.py \
    3.0 2.0 1.5 \
    /path/to/checkpoints/policy/policy_step_140000000

# Arguments:
#   3.0 = target X position (meters)
#   2.0 = target Y position (meters) 
#   1.5 = target Z altitude (meters)
#   policy_step_140000000 = your trained policy directory
```

**Expected behavior:**
- Drone takes off and navigates toward target position
- Should see velocity commands in terminal
- May not be perfect (that's why we need residuals!)

**If it crashes or doesn't move:**
- Check observation shape is (30,)
- Verify coordinate frame (Y=LEFT in ROS vs Y=RIGHT in training)
- Ensure policy loaded correctly

---

## Step 3: Collect Residual Data

Collect 3-5 flights with different trajectories to capture diverse dynamics.

**Flight 1: Aggressive Maneuvers**
```bash
python3 model_controller_with_residuals.py \
    5.0 5.0 2.5 \
    /path/to/policy_step_140000000 \
    --collect-residuals

# Saves: residual_data_YYYYMMDD_HHMMSS.pkl
```

**Flight 2: Slow Flight**
```bash
python3 model_controller_with_residuals.py \
    2.0 2.0 1.5 \
    /path/to/policy_step_140000000 \
    --collect-residuals
```

**Flight 3: Different Direction**
```bash
python3 model_controller_with_residuals.py \
    -3.0 4.0 2.0 \
    /path/to/policy_step_140000000 \
    --collect-residuals
```

**Flight 4-5: Additional Coverage (Optional)**
```bash
python3 model_controller_with_residuals.py \
    4.0 -3.0 3.0 \
    /path/to/policy_step_140000000 \
    --collect-residuals
```

**Expected Output Per Flight:**
```
[INFO] ✓ Ground truth ready
[INFO] Starting residual data collection...
[INFO] Distance to goal: 5.23m | Velocity: [1.2, 0.8, 0.0] m/s
[INFO] Samples collected: 150
...
[INFO] ============================================================
[INFO] CONTROL LOOP TIMING SUMMARY
[INFO] ============================================================
[INFO] Target rate: 10.0 Hz
[INFO] Actual rate: 10.2 Hz
[INFO] Average loop time: 98.1 ms
[INFO] Slow loops (>1.5x target): 2/590
[INFO] ============================================================
[INFO] ✓ Residual data saved: residual_data_20241210_143022.pkl
[INFO]   Samples: 590
[INFO]   Duration: 58.8 seconds
[INFO]   Rate: 10.0 Hz
```

**Good Data Indicators:**
- ✅ Actual rate: 8-12 Hz
- ✅ Sample count: 400-600 per flight
- ✅ Slow loops: < 5%
- ✅ Duration: 40-60 seconds

**If collection is slow (< 5 Hz):**
- Close other heavy programs
- Reduce control rate in script (10 Hz → 5 Hz)
- Check CPU/GPU usage

---

## Step 4: Diagnose Data Quality

Check each collected dataset for issues before fitting models.

```bash
cd /path/to/project

# Check first dataset
python3 diagnose_residual_data.py residual_data_20241210_143022.pkl

# Check all datasets
for file in residual_data_*.pkl; do
    echo "Checking $file..."
    python3 diagnose_residual_data.py "$file"
done
```

**Look For:**
- Consistent sample rate (8-12 Hz) ✅
- No large time gaps (< 100ms) ✅
- Max acceleration < 20 m/s² ✅
- No sensor glitches ✅

**Red Flags:**
- ❌ Sample rate < 5 Hz → Control loop too slow
- ❌ Max acceleration > 30 m/s² → Sensor glitches
- ❌ Large time gaps → Stuttering issues

---

## Step 5: Find Optimal Tau Parameter

The `tau` parameter represents velocity time constant. Finding the right value is critical.

**Test Multiple Tau Values:**
```bash
cd /path/to/project

# Use your best dataset (highest sample count, no gaps)
DATAFILE="residual_data_20241210_143022.pkl"

# Try range of tau values
python3 fit_dynamics_residual.py $DATAFILE --tau 0.5
python3 fit_dynamics_residual.py $DATAFILE --tau 0.8
python3 fit_dynamics_residual.py $DATAFILE --tau 1.0
python3 fit_dynamics_residual.py $DATAFILE --tau 1.2
python3 fit_dynamics_residual.py $DATAFILE --tau 1.5
python3 fit_dynamics_residual.py $DATAFILE --tau 2.0
```

**Compare Results:**

Example output for tau=1.0 (optimal):
```
⚠️  Filtered 3/295 outliers
Kept 292 samples  ← Most samples kept!
Mean residual: [0.29, -0.36, -0.08] m/s²  ← Near zero!
Cross-validation RMSE: 2.834 (+/- 0.290)  ← Lowest error!

✓ Model saved: dynamics_residual_model.pkl
```

**Select Best Tau:**
- ✅ Most samples kept (> 250, ideally > 90%)
- ✅ Lowest cross-validation RMSE
- ✅ Mean residual near zero (< 0.5 m/s² per axis)

**Typical Results:**
- tau=0.5: Too aggressive, many outliers filtered
- **tau=1.0: Usually optimal for Gazebo PX4**
- tau=2.0: Too slow, poor fit

---

## Step 6: Fit Final Residual Model

Use the best tau value to fit the final model.

```bash
# Using optimal tau from Step 5
python3 fit_dynamics_residual.py \
    residual_data_20241210_143022.pkl \
    --tau 1.0
```

**Output:**
```
======================================================================
RESIDUAL STATISTICS
======================================================================

Residual Acceleration (m/s²):
  Mean: [0.29, -0.36, -0.08]
  Std:  [1.23, 1.45, 0.89]
  Max:  [4.12, 5.23, 3.45]

Velocity range (m/s):
  X: [-3.45, 4.23]
  Y: [-2.87, 3.91]
  Z: [-1.23, 2.01]

======================================================================
FITTING KNN MODEL
======================================================================

Performing 5-fold cross-validation...
Cross-validation RMSE: 2.834 (+/- 0.290)

Training RMSE per axis:
  X: 1.234 m/s²
  Y: 1.456 m/s²
  Z: 0.987 m/s²
  Total: 1.226 m/s²

✓ Model saved: dynamics_residual_model.pkl
  Samples used: 292
  Coordinate frame: ROS (Y=LEFT, post-flip)

======================================================================
NEXT STEPS
======================================================================

1. Use this model for fine-tuning:
   model_file = 'dynamics_residual_model.pkl'
   
2. Update your simulator to use residuals:
   env = ResidualAugmentedSimulator(
       base_env_config={...},
       residual_model_path='dynamics_residual_model.pkl'
   )
   
3. Fine-tune for 20M steps with augmented sim

✓ Visualization saved: residual_analysis.png
```

**Verify Model Quality:**
```bash
# Check visualization
xdg-open residual_analysis.png
```

**Look for:**
- Residuals centered near zero (no bias)
- Predicted vs actual scatter plots align well (R² > 0.7)
- No obvious patterns in residuals (should look random)

---

## Step 7: Fine-Tune with Residual Model (Optional)

Now that you have the residual model, you can fine-tune your policy to work better in Gazebo.

**Update Simulator:**
```python
# In simulator.py or create residual_simulator.py
import pickle

class ResidualAugmentedSimulator(ImprovedCopterSimulator):
    def __init__(self, residual_model_path, **kwargs):
        super().__init__(**kwargs)
        
        # Load residual model
        with open(residual_model_path, 'rb') as f:
            residual_data = pickle.load(f)
        
        self.dynamics_knn = residual_data['knn_model']
    
    def _physics_step(self, velocity_setpoint_world):
        # Standard physics...
        
        # Add dynamics residual
        state_features = np.concatenate([
            self.velocity,
            velocity_setpoint_world
        ])
        residual_accel = self.dynamics_knn.predict([state_features])[0]
        
        # Apply residual
        total_accel += residual_accel
        
        # Continue physics...
```

**Fine-Tune Training:**
```bash
# Continue training from 140M with residuals
python train.py \
    --total-steps 160000000 \
    --checkpoint-freq 1000000 \
    --num-envs 20 \
    --use-gimbal \
    --residual-model dynamics_residual_model.pkl
```

**Mixed Training (Recommended):**
- 70% episodes: Residual-augmented simulator (Gazebo-like)
- 30% episodes: Original simulator (avoid forgetting)

---

## Quick Reference

**Complete Workflow:**
```bash
# 1. Launch Gazebo
roslaunch clover_simulation simulator.launch
roslaunch clover_simulation aruco.launch

# 2. Test policy (optional)
python3 model_controller_with_residuals.py 3.0 2.0 1.5 /path/to/policy

# 3. Collect data (3-5 flights)
python3 model_controller_with_residuals.py 5.0 5.0 2.5 /path/to/policy --collect-residuals
python3 model_controller_with_residuals.py 2.0 2.0 1.5 /path/to/policy --collect-residuals
python3 model_controller_with_residuals.py -3.0 4.0 2.0 /path/to/policy --collect-residuals

# 4. Diagnose data
python3 diagnose_residual_data.py residual_data_*.pkl

# 5. Find best tau
python3 fit_dynamics_residual.py residual_data_20241210_143022.pkl --tau 1.0

# 6. Check results
xdg-open residual_analysis.png

# 7. Fine-tune (optional)
python train.py --total-steps 160000000 --residual-model dynamics_residual_model.pkl
```

---

## Troubleshooting

**Problem: Ground truth not available**
```bash
# Check Gazebo is running
rostopic echo /gazebo/model_states --noarr

# If nothing appears:
rosservice call /gazebo/unpause_physics
```

**Problem: Control rate too slow (< 5 Hz)**
```python
# Edit model_controller_with_residuals.py
self.control_rate = 5.0  # Reduce from 10.0
```

**Problem: Very high residuals (> 10 m/s²)**
- Try larger tau values (1.0 → 2.0)
- Check for sensor glitches in diagnostics
- Collect more diverse data

**Problem: Poor model fit (RMSE > 5.0)**
- Collect more flights (5-10 instead of 3)
- Try different tau values more systematically
- Check for outliers in diagnostics

---

## Expected Results

**Good Residual Model:**
- ✅ Samples kept: > 250 (> 85%)
- ✅ Mean residual: < 0.5 m/s² per axis
- ✅ Cross-validation RMSE: < 3.0 m/s²
- ✅ R² score: > 0.7

**After Fine-Tuning (20M steps):**
- Gazebo lap time: ~10-15% faster
- Smoother velocity tracking
- More consistent gate passage
- Better handling of PX4 dynamics

---

## Files Generated

```
residual_data_20241210_143022.pkl  # Raw residual data (Flight 1)
residual_data_20241210_143155.pkl  # Raw residual data (Flight 2)
residual_data_20241210_143312.pkl  # Raw residual data (Flight 3)
residual_data_diagnostics.png      # Diagnostic plots
residual_analysis.png              # Model fit visualization
dynamics_residual_model.pkl        # Final KNN model (USE THIS)
```

**Key File:** `dynamics_residual_model.pkl` - This is what you need for fine-tuning!

---

## Summary

This process bridges the gap between your lightweight training simulator and realistic Gazebo PX4 physics:

1. ✅ Train policy to 100M+ steps in lightweight sim
2. ✅ Deploy in Gazebo, collect residual data (3-5 flights, ~3-5 minutes)
3. ✅ Find optimal tau parameter (try 0.5-2.0 range)
4. ✅ Fit KNN residual model (k=5 neighbors)
5. ✅ Fine-tune policy with augmented simulator (optional 20M steps)
6. ✅ Deploy fine-tuned policy back in Gazebo for validation

**Total Time:** ~1-2 hours (data collection + processing)

**Next Steps:** Use `dynamics_residual_model.pkl` for fine-tuning or deploy directly in Gazebo for testing!

---

**Version:** 1.0  
**Last Updated:** December 10, 2024
