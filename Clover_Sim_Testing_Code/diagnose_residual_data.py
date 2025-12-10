#!/usr/bin/env python3
"""
Diagnose issues with collected residual data.
Checks timestamps, sample rates, and data quality.
"""

import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt

def load_and_diagnose(filename):
    """Load data and run diagnostics"""
    
    print("="*70)
    print("RESIDUAL DATA DIAGNOSTICS")
    print("="*70)
    print(f"\nLoading: {filename}\n")
    
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    timestamps = np.array(data['timestamp'])
    positions = np.array(data['position'])
    velocities = np.array(data['velocity'])
    cmd_velocities = np.array(data['commanded_velocity'])
    
    n_samples = len(timestamps)
    
    # Basic stats
    print("SAMPLE STATISTICS")
    print("-"*70)
    print(f"Total samples: {n_samples}")
    print(f"Duration: {timestamps[-1] - timestamps[0]:.2f} seconds")
    
    # Timestamp analysis
    dt_values = np.diff(timestamps)
    avg_dt = dt_values.mean()
    avg_rate = 1.0 / avg_dt if avg_dt > 0 else 0
    
    print(f"\nTIMING ANALYSIS")
    print("-"*70)
    print(f"Average dt: {avg_dt*1000:.2f} ms")
    print(f"Average rate: {avg_rate:.1f} Hz")
    print(f"Min dt: {dt_values.min()*1000:.2f} ms")
    print(f"Max dt: {dt_values.max()*1000:.2f} ms")
    print(f"Std dt: {dt_values.std()*1000:.2f} ms")
    
    # Check for problems
    print(f"\nPROBLEM DETECTION")
    print("-"*70)
    
    # Large gaps
    large_gaps = dt_values > 0.1  # >100ms gap
    if large_gaps.any():
        print(f"⚠️  Found {large_gaps.sum()} large time gaps (>100ms)")
        print(f"   Largest gap: {dt_values.max()*1000:.1f} ms")
        print(f"   This suggests control loop stalling or processing delays")
    else:
        print("✓ No large time gaps detected")
    
    # Very small dt (duplicate timestamps or clock issues)
    small_dt = dt_values < 0.001  # <1ms
    if small_dt.any():
        print(f"⚠️  Found {small_dt.sum()} very small dt values (<1ms)")
        print(f"   Smallest: {dt_values.min()*1000:.3f} ms")
    else:
        print("✓ No clock issues detected")
    
    # Data range analysis
    print(f"\nDATA RANGES")
    print("-"*70)
    
    speeds = np.linalg.norm(velocities, axis=1)
    cmd_speeds = np.linalg.norm(cmd_velocities, axis=1)
    
    print(f"Actual velocity:")
    print(f"  X: [{velocities[:, 0].min():.2f}, {velocities[:, 0].max():.2f}] m/s")
    print(f"  Y: [{velocities[:, 1].min():.2f}, {velocities[:, 1].max():.2f}] m/s")
    print(f"  Z: [{velocities[:, 2].min():.2f}, {velocities[:, 2].max():.2f}] m/s")
    print(f"  Speed: [{speeds.min():.2f}, {speeds.max():.2f}] m/s")
    
    print(f"\nCommanded velocity:")
    print(f"  X: [{cmd_velocities[:, 0].min():.2f}, {cmd_velocities[:, 0].max():.2f}] m/s")
    print(f"  Y: [{cmd_velocities[:, 1].min():.2f}, {cmd_velocities[:, 1].max():.2f}] m/s")
    print(f"  Z: [{cmd_velocities[:, 2].min():.2f}, {cmd_velocities[:, 2].max():.2f}] m/s")
    print(f"  Speed: [{cmd_speeds.min():.2f}, {cmd_speeds.max():.2f}] m/s")
    
    # Compute actual accelerations
    actual_accels = np.diff(velocities, axis=0) / dt_values[:, np.newaxis]
    accel_mags = np.linalg.norm(actual_accels, axis=1)
    
    print(f"\nActual accelerations (from velocity changes):")
    print(f"  X: [{actual_accels[:, 0].min():.2f}, {actual_accels[:, 0].max():.2f}] m/s²")
    print(f"  Y: [{actual_accels[:, 1].min():.2f}, {actual_accels[:, 1].max():.2f}] m/s²")
    print(f"  Z: [{actual_accels[:, 2].min():.2f}, {actual_accels[:, 2].max():.2f}] m/s²")
    print(f"  Magnitude: [{accel_mags.min():.2f}, {accel_mags.max():.2f}] m/s²")
    
    # Check for unrealistic accelerations
    if accel_mags.max() > 20:
        print(f"\n⚠️  WARNING: Very high accelerations detected!")
        print(f"   Max acceleration: {accel_mags.max():.1f} m/s²")
        print(f"   This is unrealistic for a quadrotor (normal: <10 m/s²)")
        print(f"   Likely causes:")
        print(f"   - Timestamp jumps causing large dt")
        print(f"   - Data corruption")
        print(f"   - Velocity estimates jumping (sensor glitches)")
        
        # Find where the spikes occur
        spike_indices = np.where(accel_mags > 20)[0]
        print(f"\n   Spike locations (first 5):")
        for idx in spike_indices[:5]:
            print(f"   Sample {idx}: dt={dt_values[idx]*1000:.1f}ms, "
                  f"accel={accel_mags[idx]:.1f} m/s²")
    
    # Check position drift
    pos_drift = np.linalg.norm(positions[-1] - positions[0])
    print(f"\nPosition drift: {pos_drift:.2f} m")
    
    # Recommendations
    print(f"\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    if avg_rate < 30:
        print("⚠️  Sample rate too low (<30 Hz)")
        print("   → Check control loop frequency")
        print("   → Look for processing bottlenecks")
        print("   → Reduce control rate if needed (set to 20-30 Hz)")
    
    if large_gaps.any():
        print("⚠️  Large timing gaps detected")
        print("   → Control loop may be stalling")
        print("   → Check for expensive operations in loop")
        print("   → Consider running fewer parallel environments")
    
    if accel_mags.max() > 20:
        print("⚠️  Unrealistic accelerations")
        print("   → Filter outliers before fitting KNN")
        print("   → Check for sensor glitches")
        print("   → Verify ground truth data quality")
    
    if n_samples < 1000:
        print("⚠️  Too few samples")
        print(f"   → Got {n_samples}, need at least 1500-2000")
        print("   → Increase flight duration")
        print("   → Fix control loop rate issues first")
    
    # Create visualizations
    create_diagnostics_plot(timestamps, velocities, cmd_velocities, 
                           actual_accels, dt_values)
    
    return data

def create_diagnostics_plot(timestamps, velocities, cmd_velocities, 
                            actual_accels, dt_values):
    """Create diagnostic plots"""
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # Plot 1: Timing
    ax = axes[0, 0]
    ax.plot(dt_values * 1000, 'b-', alpha=0.7, linewidth=0.5)
    ax.axhline(y=20, color='g', linestyle='--', label='Target: 50Hz (20ms)')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('dt (ms)')
    ax.set_title('Time Between Samples')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Histogram of dt
    ax = axes[0, 1]
    ax.hist(dt_values * 1000, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=20, color='g', linestyle='--', label='Target: 20ms')
    ax.set_xlabel('dt (ms)')
    ax.set_ylabel('Count')
    ax.set_title('dt Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Velocity tracking
    ax = axes[1, 0]
    time_rel = timestamps - timestamps[0]
    ax.plot(time_rel, velocities[:, 0], 'b-', label='Actual Vx', linewidth=1)
    ax.plot(time_rel, cmd_velocities[:, 0], 'r--', label='Cmd Vx', linewidth=1, alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity X (m/s)')
    ax.set_title('X Velocity Tracking')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Speed
    ax = axes[1, 1]
    speeds = np.linalg.norm(velocities, axis=1)
    cmd_speeds = np.linalg.norm(cmd_velocities, axis=1)
    ax.plot(time_rel, speeds, 'b-', label='Actual Speed', linewidth=1)
    ax.plot(time_rel, cmd_speeds, 'r--', label='Cmd Speed', linewidth=1, alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Speed Tracking')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Accelerations
    ax = axes[2, 0]
    time_rel_accel = time_rel[1:]
    ax.plot(time_rel_accel, actual_accels[:, 0], 'b-', label='Accel X', linewidth=0.5)
    ax.plot(time_rel_accel, actual_accels[:, 1], 'g-', label='Accel Y', linewidth=0.5)
    ax.plot(time_rel_accel, actual_accels[:, 2], 'r-', label='Accel Z', linewidth=0.5)
    ax.axhline(y=20, color='k', linestyle='--', alpha=0.5, label='±20 m/s² threshold')
    ax.axhline(y=-20, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration (m/s²)')
    ax.set_title('Actual Accelerations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Acceleration magnitude
    ax = axes[2, 1]
    accel_mags = np.linalg.norm(actual_accels, axis=1)
    ax.plot(time_rel_accel, accel_mags, 'b-', linewidth=0.5)
    ax.axhline(y=10, color='g', linestyle='--', label='Normal max (~10 m/s²)')
    ax.axhline(y=20, color='orange', linestyle='--', label='High (20 m/s²)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration Magnitude (m/s²)')
    ax.set_title('Acceleration Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, min(accel_mags.max() * 1.1, 50)])
    
    plt.tight_layout()
    plt.savefig('residual_data_diagnostics.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Diagnostic plots saved: residual_data_diagnostics.png")
    
    try:
        plt.show()
    except:
        pass

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 diagnose_residual_data.py <residual_data.pkl>")
        print("\nExample:")
        print("  python3 diagnose_residual_data.py residual_data_20241123_212258.pkl")
        sys.exit(1)
    
    filename = sys.argv[1]
    data = load_and_diagnose(filename)
