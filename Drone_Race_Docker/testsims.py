"""
Comprehensive Verification Tests for Clover Simulator Arithmetic
Run these after defining the simulator to verify geometric correctness
"""

print("\n" + "="*70)
print("COMPREHENSIVE SIMULATOR ARITHMETIC VERIFICATION")
print("="*70)
print("Running all validation tests from v3 Cell 5b...")
print()

# Create test simulator directly (bypass TF-Agents wrapper for detailed testing)
from dataclasses import dataclass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import tensorflow as tf

from src.tf_agents_wrapper import TFAgentsCopterEnv
from src.simulator import ImprovedCopterSimulator, CloverSpecs
from tf_agents.environments import parallel_py_environment, tf_py_environment


# Create test environment
test_sim = ImprovedCopterSimulator(
    specs=CloverSpecs(),
    dt=0.004,
    domain_randomization=False,  # Disable for deterministic testing
    add_noise=False
)

def test_gate_geometry(env):
    """Test 1: Verify gate corners form proper squares"""
    print("="*60)
    print("TEST 1: Gate Geometry Verification")
    print("="*60)

    for gate_idx in range(env.num_gates):
        corners = env._get_gate_corners(gate_idx)

        # Test 1a: All sides should be 1.0m
        sides = [
            np.linalg.norm(corners[0] - corners[1]),  # Top edge
            np.linalg.norm(corners[1] - corners[2]),  # Left edge
            np.linalg.norm(corners[2] - corners[3]),  # Bottom edge
            np.linalg.norm(corners[3] - corners[0]),  # Right edge
        ]

        # Test 1b: Diagonals should be sqrt(2) ≈ 1.414m
        diag1 = np.linalg.norm(corners[0] - corners[2])
        diag2 = np.linalg.norm(corners[1] - corners[3])

        # Test 1c: Corners should be coplanar
        v1 = corners[1] - corners[0]
        v2 = corners[2] - corners[0]
        v3 = corners[3] - corners[0]
        normal = np.cross(v1, v2)
        normal = normal / (np.linalg.norm(normal) + 1e-6)
        coplanar_error = abs(np.dot(normal, v3))

        print(f"\nGate {gate_idx}:")
        print(f"  Sides: {sides[0]:.4f}, {sides[1]:.4f}, {sides[2]:.4f}, {sides[3]:.4f} m")
        print(f"  Expected: 1.0000 m each")
        print(f"  Diagonals: {diag1:.4f}, {diag2:.4f} m")
        print(f"  Expected: 1.4142 m")
        print(f"  Coplanar error: {coplanar_error:.6f}")
        print(f"  Expected: ~0.000000")

        # Assertions
        assert all(abs(s - 1.0) < 0.001 for s in sides), f"Gate {gate_idx} sides incorrect!"
        assert abs(diag1 - np.sqrt(2)) < 0.001, f"Gate {gate_idx} diagonal 1 incorrect!"
        assert abs(diag2 - np.sqrt(2)) < 0.001, f"Gate {gate_idx} diagonal 2 incorrect!"
        assert coplanar_error < 1e-6, f"Gate {gate_idx} corners not coplanar!"

    print("\n✅ All gates pass geometry tests!")


def test_gate_orientation(env):
    """Test 2: Verify gates face toward next gate"""
    print("\n" + "="*60)
    print("TEST 2: Gate Orientation Verification")
    print("="*60)

    for gate_idx in range(env.num_gates):
        gate_center = env.gates[gate_idx]
        next_idx = (gate_idx + 1) % env.num_gates

        # Calculate expected gate normal
        to_next = env.gates[next_idx] - gate_center
        expected_normal = to_next / (np.linalg.norm(to_next) + 1e-6)

        # Calculate actual gate normal from corners
        corners = env._get_gate_corners(gate_idx)
        v1 = corners[1] - corners[0]  # Top left to top right
        v2 = corners[2] - corners[1]  # Top right to bottom right
        actual_normal = np.cross(v1, v2)
        actual_normal = actual_normal / (np.linalg.norm(actual_normal) + 1e-6)

        # Normals should be parallel (dot product close to ±1)
        alignment = abs(np.dot(expected_normal, actual_normal))

        print(f"\nGate {gate_idx} → Gate {next_idx}:")
        print(f"  Expected normal: [{expected_normal[0]:+.3f}, {expected_normal[1]:+.3f}, {expected_normal[2]:+.3f}]")
        print(f"  Actual normal:   [{actual_normal[0]:+.3f}, {actual_normal[1]:+.3f}, {actual_normal[2]:+.3f}]")
        print(f"  Alignment: {alignment:.6f} (should be ~1.0)")

        assert alignment > 0.99, f"Gate {gate_idx} not facing next gate!"

    print("\n✅ All gates correctly oriented!")


def test_gate_crossing_detection(env):
    """Test 3: Verify gate crossing detection"""
    print("\n" + "="*60)
    print("TEST 3: Gate Crossing Detection")
    print("="*60)

    test_cases = [
        # (start_key, end_key, expected, description)
        ("before_gate", "after_gate", True, "Center pass-through"),
        ("before_gate", "corner", True, "Corner pass-through"),
        ("before_gate", "outside", False, "Miss to the side"),
        ("after_gate", "before_gate", False, "Wrong direction"),
        ("parallel", "parallel", False, "Parallel flight (no crossing)"),
    ]

    for gate_idx in [0]:  # Test with first gate
        gate_center = env.gates[gate_idx]
        next_idx = (gate_idx + 1) % env.num_gates
        gate_normal = env.gates[next_idx] - gate_center
        gate_normal = gate_normal / (np.linalg.norm(gate_normal) + 1e-6)

        # Create positions for each test case
        positions = {
            "before_gate": gate_center - 1.0 * gate_normal,  # 1m before
            "after_gate": gate_center + 1.0 * gate_normal,   # 1m after
            "corner": gate_center + 1.0 * gate_normal + np.array([0.4, 0.0, 0.4]),  # Through corner
            "outside": gate_center + 1.0 * gate_normal + np.array([0.8, 0.0, 0.0]),  # Outside bounds
            "parallel": gate_center + np.array([0.0, 1.0, 0.0]),  # Parallel to plane
        }

        print(f"\nTesting Gate {gate_idx}:")

        for start_key, end_key, expected, description in test_cases:
            # Set up environment
            env.reset()
            env.current_gate_idx = gate_idx

            start_pos = positions[start_key]
            end_pos = positions[end_key]

            # Simulate movement
            env.position = start_pos.copy()
            env.velocity = (end_pos - start_pos) / env.control_dt
            env.position = end_pos.copy()

            detected = env._passed_through_gate()

            status = "✅" if detected == expected else "❌"
            print(f"  {status} {description}: detected={detected}, expected={expected}")

            assert detected == expected, f"Gate crossing detection failed for {description}"

    print("\n✅ Gate crossing detection tests complete!")


def test_camera_angle_calculation(env):
    """Test 4: Verify camera pointing angle"""
    print("\n" + "="*60)
    print("TEST 4: Camera Angle Calculation")
    print("="*60)

    env.reset()
    env.current_gate_idx = 0
    gate_center = env.gates[0]

    if not env.use_gimbal:
        test_cases = [
            # (drone_pos, orientation_euler, expected_angle_deg, description)
            (gate_center - np.array([2, 0, 0]), [0, 0, 0], 0, "Facing gate directly"),
            (gate_center - np.array([2, 0, 0]), [0, np.pi/4, 0], 45, "45° pitch up"),
            (gate_center - np.array([2, 0, 0]), [0, -np.pi/4, 0], 45, "45° pitch down"),
            (gate_center - np.array([2, 0, 0]), [0, 0, np.pi/2], 90, "90° yaw (perpendicular)"),
            (gate_center - np.array([2, 0, 0]), [0, 0, np.pi], 180, "Facing away"),
        ]
    else:
        test_cases = [
            # (drone_pos, orientation_euler, expected_angle_deg, description)
            (gate_center - np.array([2, 0, 0]), [0, 0, 0], 0, "Facing gate directly"),
            (gate_center - np.array([2, 0, 0]), [0, np.pi/4, 0], 0, "45° pitch up"),
            (gate_center - np.array([2, 0, 0]), [0, -np.pi/4, 0], 0, "45° pitch down"),
            (gate_center - np.array([2, 0, 0]), [0, 0, np.pi/2], 90, "90° yaw (perpendicular)"),
            (gate_center - np.array([2, 0, 0]), [0, 0, np.pi], 180, "Facing away"),
        ]

    for drone_pos, euler, expected_deg, description in test_cases:
        env.position = drone_pos.copy()
        env.orientation = env._euler_to_rotation(euler[0], euler[1], euler[2])

        angle_rad = env._get_camera_pointing_angle()
        angle_deg = np.degrees(angle_rad)

        error = abs(angle_deg - expected_deg)
        status = "✅" if error < 5 else "❌"  # 5° tolerance

        print(f"  {status} {description}")
        print(f"     Calculated: {angle_deg:.1f}°, Expected: {expected_deg:.1f}°, Error: {error:.1f}°")

        assert error < 5, f"Camera angle error too large: {error:.1f}°"

    print("\n✅ Camera angle calculations correct!")


def test_reward_components(env):
    """Test 5: Verify reward function components"""
    print("\n" + "="*60)
    print("TEST 5: Reward Function Components")
    print("="*60)

    env.reset()
    env.current_gate_idx = 0
    gate_center = env.gates[0]

    # Test r_progress
    print("\n5a. Testing r_progress:")

    # Moving closer should give positive reward
    env.position = gate_center - np.array([5, 0, 0])
    env.velocity = np.array([2, 0, 0])  # Moving toward gate
    env.previous_action = np.zeros(3)
    env.last_action = np.zeros(3)

    reward, _, _, info = env._compute_reward_and_termination()
    r_prog = info['r_progress']

    print(f"  Moving toward gate: r_progress = {r_prog:.4f}")
    assert r_prog > 0, "r_progress should be positive when approaching gate!"

    # Moving away should give negative reward
    env.velocity = np.array([-2, 0, 0])  # Moving away
    reward, _, _, info = env._compute_reward_and_termination()
    r_prog = info['r_progress']

    print(f"  Moving away from gate: r_progress = {r_prog:.4f}")
    assert r_prog < 0, "r_progress should be negative when leaving gate!"

    # Test r_perc
    print("\n5b. Testing r_perc:")

    # Camera aligned with gate
    env.position = gate_center - np.array([2, 0, 0])
    env.orientation = np.eye(3)  # Facing +X (toward gate)
    env.velocity = np.zeros(3)

    reward, _, _, info = env._compute_reward_and_termination()
    r_perc_aligned = info['r_perc']
    camera_angle = info['camera_angle_deg']

    print(f"  Camera aligned (angle={camera_angle:.1f}°): r_perc = {r_perc_aligned:.4f}")

    # Camera perpendicular to gate
    env.orientation = env._euler_to_rotation(0, 0, np.pi/2)  # Facing +Y
    reward, _, _, info = env._compute_reward_and_termination()
    r_perc_perp = info['r_perc']
    camera_angle = info['camera_angle_deg']

    print(f"  Camera perpendicular (angle={camera_angle:.1f}°): r_perc = {r_perc_perp:.4f}")
    assert r_perc_aligned > r_perc_perp, "Aligned camera should have higher r_perc!"

    # Test r_smooth
    print("\n5c. Testing r_smooth:")

    # Smooth action
    env.previous_action = np.array([0.5, 0.0, 0.0])
    env.last_action = np.array([0.5, 0.0, 0.0])
    reward, _, _, info = env._compute_reward_and_termination()
    r_smooth_stable = info['r_smooth']

    print(f"  No action change: r_smooth = {r_smooth_stable:.6f}")

    # Large action change
    env.last_action = np.array([-0.5, 0.0, 0.0])
    reward, _, _, info = env._compute_reward_and_termination()
    r_smooth_jerky = info['r_smooth']

    print(f"  Large action change: r_smooth = {r_smooth_jerky:.6f}")
    assert r_smooth_jerky < r_smooth_stable, "Jerky actions should have lower r_smooth!"

    print("\n✅ Reward components functioning correctly!")


# ============================================================================
# RUN ALL VALIDATION TESTS
# ============================================================================

try:
    test_gate_geometry(test_sim)
    test_gate_orientation(test_sim)
    test_gate_crossing_detection(test_sim)
    test_camera_angle_calculation(test_sim)
    test_reward_components(test_sim)

    print("\n" + "="*70)
    print("✅ ALL SIMULATOR ARITHMETIC TESTS PASSED!")
    print("="*70)
    print("\nSimulator arithmetic is mathematically correct.")
    print("You can proceed with confidence to TF-Agents training.\n")

except AssertionError as e:
    print(f"\n❌ TEST FAILED: {e}")
    print("Please review the arithmetic logic.")
    raise

def validate_wrapper_after_fixes():
    """
    Verify that TF-Agents wrapper works correctly with fixed simulator.

    This test checks:
    1. Action interpretation (world frame, not body frame)
    2. Forward motion results in positive X displacement
    3. Episode length enforcement (1500 steps)
    4. Observation shape and validity
    """
    print("\n" + "="*70)
    print("VALIDATING TF-AGENTS WRAPPER AFTER PHYSICS FIXES")
    print("="*70)

    env = TFAgentsCopterEnv(seed=42, episode_length=1500)

    # Test 1: Reset
    print("\n[TEST 1] Reset")
    time_step = env.reset()
    assert time_step.is_first(), "First timestep should be FIRST"
    assert time_step.observation.shape == (30,), "Observation shape mismatch"
    print("✅ Reset works")

    # Test 2: Forward action produces forward motion
    print("\n[TEST 2] Forward Action → Forward Motion")

    # Reset to known state
    time_step = env.reset()
    initial_pos = env._sim.position.copy()

    # Execute forward action for 50 steps
    action = np.array([0.6, 0.0, 0.0], dtype=np.float32)  # Forward

    for i in range(50):
        time_step = env.step(action)
        if time_step.is_last():
            print(f"❌ Episode ended early at step {i}")
            break

    final_pos = env._sim.position.copy()
    displacement_x = final_pos[0] - initial_pos[0]

    print(f"  Initial X: {initial_pos[0]:.2f}m")
    print(f"  Final X:   {final_pos[0]:.2f}m")
    print(f"  Displacement: {displacement_x:.2f}m")

    if displacement_x > 0.5:
        print("✅ Forward action produces forward motion")
    else:
        print(f"❌ Forward action produces: {displacement_x:.2f}m (expected >0.5m)")
        print("   Check that physics fixes are applied!")

    # Test 3: Episode length
    print("\n[TEST 3] Episode Length (1500 steps)")

    env.reset()
    step_count = 0

    # Gentle action to avoid crashes
    action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    while step_count < 1600:  # Safety limit
        time_step = env.step(action)
        step_count += 1

        if time_step.is_last():
            break

    print(f"  Episode length: {step_count} steps")

    if step_count == 1500:
        print("✅ Episode length correct (1500 steps)")
    elif step_count < 1500:
        print(f"⚠️  Episode ended early at {step_count} steps (likely crash)")
    else:
        print(f"❌ Episode exceeded 1500 steps: {step_count}")

    # Test 4: Observation validity
    print("\n[TEST 4] Observation Validity")

    env.reset()

    for _ in range(10):
        action = np.random.uniform(-0.5, 0.5, size=3).astype(np.float32)
        time_step = env.step(action)

        obs = time_step.observation

        if np.any(np.isnan(obs)):
            print("❌ NaN detected in observation!")
            break
        if np.any(np.isinf(obs)):
            print("❌ Inf detected in observation!")
            break
    else:
        print("✅ No NaN/Inf in observations")

    # Test 5: Reward sanity
    print("\n[TEST 5] Reward Sanity")

    env.reset()
    rewards = []

    for _ in range(100):
        action = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        time_step = env.step(action)
        rewards.append(float(time_step.reward))

        if time_step.is_last():
            break

    print(f"  Rewards: min={min(rewards):.2f}, max={max(rewards):.2f}, "
          f"mean={np.mean(rewards):.2f}")

    if all(np.isfinite(r) for r in rewards):
        print("✅ All rewards finite")
    else:
        print("❌ Non-finite rewards detected!")

    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)

validate_wrapper_after_fixes()

print("="*70)
print("VALIDATION - TF Environment Wrapper")
print("="*70)

# Create TF-wrapped environment (single)
py_env = TFAgentsCopterEnv(seed=42)
tf_env_single = tf_py_environment.TFPyEnvironment(py_env)

print("\n[TEST 9] TF Environment Properties")
print("-" * 70)

print(f"✓ Batch size: {tf_env_single.batch_size}")
print(f"✓ Batched: {tf_env_single.batched}")
print(f"✓ Time step spec: {tf_env_single.time_step_spec()}")

# Test reset
time_step = tf_env_single.reset()
print(f"✓ TF reset successful")
print(f"  Observation dtype: {time_step.observation.dtype}")
print(f"  Observation device: {time_step.observation.device}")

# Test step
action = tf.constant([[0.0, 0.0, 0.0]], dtype=tf.float32)
time_step = tf_env_single.step(action)
print(f"✓ TF step successful")
print(f"  Reward: {time_step.reward.numpy()[0]:.4f}")

print("\n" + "="*70)
print("✅ TF WRAPPER VALIDATION PASSED!")
print("="*70)
print()