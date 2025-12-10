#!/usr/bin/env python3
"""
Process collected residual data and fit KNN model.

This script:
1. Loads data collected during Gazebo flights
2. Computes residual accelerations (actual - predicted)
3. Fits KNN model with k=5 neighbors
4. Saves model for use in fine-tuning

IMPORTANT: This processes data in ROS coordinate frame (post Y-flip)
"""

import numpy as np
import pickle
import sys
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

def load_data(filename):
    """Load residual data from pickle file"""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data['timestamp'])} samples from {filename}")
    return data

def estimate_optimal_tau(data, tau_range=None):
    """
    Estimate optimal tau (velocity time constant) by grid search.
    
    The tau that minimizes residual magnitude is likely closest to reality.
    """
    if tau_range is None:
        tau_range = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1.0]
    
    print(f"\n" + "="*70)
    print("TAU PARAMETER ESTIMATION")
    print("="*70)
    print(f"Testing tau values: {tau_range}")
    
    timestamps = data['timestamp']
    velocities = data['velocity']
    cmd_velocities = data['commanded_velocity']
    
    results = []
    
    for tau in tau_range:
        sim_params = {'velocity_time_constant': tau}
        residual_sum = 0
        count = 0
        
        for i in range(len(timestamps) - 1):
            dt = timestamps[i + 1] - timestamps[i]
            if dt <= 0 or dt > 0.1:
                continue
            
            actual_accel = (velocities[i + 1] - velocities[i]) / dt
            predicted_accel = compute_simulated_acceleration(
                velocities[i], cmd_velocities[i], dt, sim_params
            )
            residual = actual_accel - predicted_accel
            residual_sum += np.linalg.norm(residual)
            count += 1
        
        avg_residual = residual_sum / count if count > 0 else float('inf')
        results.append((tau, avg_residual))
    
    # Find best tau
    results.sort(key=lambda x: x[1])
    best_tau = results[0][0]
    
    print(f"\nResults (tau, avg residual magnitude):")
    for tau, res in results:
        marker = " ← BEST" if tau == best_tau else ""
        print(f"  tau={tau:.2f}: {res:.3f} m/s²{marker}")
    
    print(f"\n✓ Recommended tau: {best_tau}")
    
    return best_tau

def compute_simulated_acceleration(velocity, cmd_velocity, dt, sim_params):
    """
    Compute what your lightweight simulator would predict.
    
    IMPORTANT: This must match your ImprovedCopterSimulator's velocity dynamics.
    Adjust the parameters to match your actual implementation.
    
    Args:
        velocity: Current velocity [vx, vy, vz]
        cmd_velocity: Commanded velocity [vx, vy, vz] (post-flip, ROS frame)
        dt: Time step
        sim_params: Dict with 'velocity_time_constant' etc.
    
    Returns:
        predicted_accel: What sim predicts [ax, ay, az]
    """
    # Example: First-order velocity response
    # Your actual sim might be more complex
    tau = sim_params.get('velocity_time_constant', 0.1)
    
    # Simple first-order dynamics: dv/dt = (cmd_vel - vel) / tau
    predicted_accel = (cmd_velocity - velocity) / tau
    
    # If your sim includes gravity compensation, add it here
    # predicted_accel[2] += 9.81  # Uncomment if needed
    
    return predicted_accel

def compute_residuals(data, sim_params):
    """
    Compute residual accelerations with outlier filtering.
    
    Residual = Actual acceleration (from ground truth) - Predicted acceleration (from sim)
    
    Returns:
        residuals: (N, 3) array of residual accelerations
        features: (N, 6) array of [vel_x, vel_y, vel_z, cmd_vel_x, cmd_vel_y, cmd_vel_z]
    """
    timestamps = data['timestamp']
    velocities = data['velocity']
    cmd_velocities = data['commanded_velocity']
    
    residuals = []
    features = []
    valid_indices = []
    
    # First pass: compute all residuals
    all_residuals = []
    all_features = []
    all_dt = []
    
    for i in range(len(timestamps) - 1):
        dt = timestamps[i + 1] - timestamps[i]
        
        # Skip bad timestamps
        if dt <= 0 or dt > 0.1:
            continue
        
        # Actual acceleration from ground truth
        actual_accel = (velocities[i + 1] - velocities[i]) / dt
        
        # Predicted acceleration from lightweight sim
        predicted_accel = compute_simulated_acceleration(
            velocities[i],
            cmd_velocities[i],
            dt,
            sim_params
        )
        
        # Residual
        residual = actual_accel - predicted_accel
        
        # Feature
        feature = np.concatenate([
            velocities[i],
            cmd_velocities[i]
        ])
        
        all_residuals.append(residual)
        all_features.append(feature)
        all_dt.append(dt)
    
    all_residuals = np.array(all_residuals)
    all_features = np.array(all_features)
    all_dt = np.array(all_dt)
    
    # Filter outliers based on acceleration magnitude
    accel_mags = np.linalg.norm(all_residuals, axis=1)
    
    # Use 99th percentile as threshold (keeps 99% of data)
    threshold = np.percentile(accel_mags, 99)
    
    # Quadrotors typically have max ~10-15 m/s² acceleration
    # If threshold is higher, cap it
    threshold = min(threshold, 15.0)
    
    valid_mask = accel_mags < threshold
    
    residuals = all_residuals[valid_mask]
    features = all_features[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    
    n_filtered = (~valid_mask).sum()
    if n_filtered > 0:
        print(f"\n⚠️  Filtered {n_filtered}/{len(all_residuals)} outliers "
              f"(>{threshold:.1f} m/s²)")
        print(f"   Kept {len(residuals)} samples for training")
    
    print(f"\nComputed {len(residuals)} residual samples (after filtering)")
    print(f"Feature shape: {features.shape}")
    print(f"Residual shape: {residuals.shape}")
    
    return residuals, features, valid_indices

def analyze_residuals(residuals, features, data, valid_indices):
    """Print statistical analysis of residuals"""
    print("\n" + "="*60)
    print("RESIDUAL STATISTICS")
    print("="*60)
    
    print(f"\nResidual Acceleration (m/s²):")
    print(f"  Mean: [{residuals[:, 0].mean():.4f}, {residuals[:, 1].mean():.4f}, "
          f"{residuals[:, 2].mean():.4f}]")
    print(f"  Std:  [{residuals[:, 0].std():.4f}, {residuals[:, 1].std():.4f}, "
          f"{residuals[:, 2].std():.4f}]")
    print(f"  Max:  [{np.abs(residuals[:, 0]).max():.4f}, "
          f"{np.abs(residuals[:, 1]).max():.4f}, "
          f"{np.abs(residuals[:, 2]).max():.4f}]")
    
    print(f"\nVelocity range (m/s):")
    velocities = features[:, :3]
    print(f"  X: [{velocities[:, 0].min():.2f}, {velocities[:, 0].max():.2f}]")
    print(f"  Y: [{velocities[:, 1].min():.2f}, {velocities[:, 1].max():.2f}]")
    print(f"  Z: [{velocities[:, 2].min():.2f}, {velocities[:, 2].max():.2f}]")
    
    print(f"\nCommand velocity range (m/s):")
    cmd_vels = features[:, 3:]
    print(f"  X: [{cmd_vels[:, 0].min():.2f}, {cmd_vels[:, 0].max():.2f}]")
    print(f"  Y: [{cmd_vels[:, 1].min():.2f}, {cmd_vels[:, 1].max():.2f}]")
    print(f"  Z: [{cmd_vels[:, 2].min():.2f}, {cmd_vels[:, 2].max():.2f}]")
    
    # Correlation analysis
    print(f"\nResidual correlation with speed:")
    speeds = np.linalg.norm(velocities, axis=1)
    for i, axis in enumerate(['X', 'Y', 'Z']):
        corr = np.corrcoef(speeds, residuals[:, i])[0, 1]
        print(f"  {axis}: {corr:.3f}")

def fit_knn_model(features, residuals, n_neighbors=5):
    """
    Fit KNN model to predict residual accelerations.
    
    Args:
        features: (N, 6) array [vel, cmd_vel]
        residuals: (N, 3) array [residual_accel_x, y, z]
        n_neighbors: Number of neighbors for KNN
    
    Returns:
        knn: Trained KNeighborsRegressor
    """
    print("\n" + "="*60)
    print("FITTING KNN MODEL")
    print("="*60)
    
    knn = KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights='distance',
        algorithm='auto',
        n_jobs=-1  # Use all CPU cores
    )
    
    # Cross-validation
    print(f"\nPerforming 5-fold cross-validation...")
    scores = cross_val_score(
        knn, features, residuals, 
        cv=5, 
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    rmse_scores = np.sqrt(-scores)
    print(f"Cross-validation RMSE: {rmse_scores.mean():.4f} "
          f"(+/- {rmse_scores.std():.4f})")
    
    # Fit on full dataset
    print(f"\nFitting on full dataset ({len(features)} samples)...")
    knn.fit(features, residuals)
    
    # Training error
    predictions = knn.predict(features)
    train_mse = np.mean((predictions - residuals) ** 2, axis=0)
    train_rmse = np.sqrt(train_mse)
    print(f"Training RMSE per axis:")
    print(f"  X: {train_rmse[0]:.4f} m/s²")
    print(f"  Y: {train_rmse[1]:.4f} m/s²")
    print(f"  Z: {train_rmse[2]:.4f} m/s²")
    print(f"  Total: {np.sqrt(np.mean(train_mse)):.4f} m/s²")
    
    return knn, predictions

def visualize_residuals(residuals, predictions, features, output_file='residual_analysis.png'):
    """Create visualization of residual model fit"""
    
    fig = plt.figure(figsize=(15, 10))
    labels = ['X', 'Y', 'Z']
    colors = ['red', 'green', 'blue']
    
    # Plot 1-3: Residual distributions
    for i in range(3):
        ax = plt.subplot(3, 3, i + 1)
        ax.hist(residuals[:, i], bins=50, alpha=0.7, color=colors[i], 
               label='Actual', density=True)
        ax.hist(predictions[:, i], bins=50, alpha=0.7, color='orange',
               label='Predicted', density=True)
        ax.set_xlabel(f'Residual Accel {labels[i]} (m/s²)')
        ax.set_ylabel('Density')
        ax.set_title(f'{labels[i]}-axis Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 4-6: Actual vs Predicted scatter
    for i in range(3):
        ax = plt.subplot(3, 3, i + 4)
        ax.scatter(residuals[:, i], predictions[:, i], 
                  alpha=0.3, s=1, color=colors[i])
        
        # Perfect fit line
        min_val = min(residuals[:, i].min(), predictions[:, i].min())
        max_val = max(residuals[:, i].max(), predictions[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 
               'r--', linewidth=2, label='Perfect fit')
        
        ax.set_xlabel(f'Actual Residual {labels[i]} (m/s²)')
        ax.set_ylabel(f'Predicted Residual {labels[i]} (m/s²)')
        ax.set_title(f'{labels[i]}-axis: Actual vs Predicted')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add R² score
        from sklearn.metrics import r2_score
        r2 = r2_score(residuals[:, i], predictions[:, i])
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', 
               transform=ax.transAxes, 
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 7-9: Residuals vs speed
    speeds = np.linalg.norm(features[:, :3], axis=1)
    for i in range(3):
        ax = plt.subplot(3, 3, i + 7)
        ax.scatter(speeds, residuals[:, i], alpha=0.3, s=1, color=colors[i])
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
        ax.set_xlabel('Speed (m/s)')
        ax.set_ylabel(f'Residual {labels[i]} (m/s²)')
        ax.set_title(f'{labels[i]}-axis vs Speed')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved: {output_file}")
    
    # Show plot if not in headless mode
    try:
        plt.show()
    except:
        pass

def save_model(knn_model, residuals, features, output_file='dynamics_residual_model.pkl'):
    """Save trained KNN model and metadata"""
    
    model_data = {
        'knn_model': knn_model,
        'feature_names': ['vel_x', 'vel_y', 'vel_z', 
                         'cmd_vel_x', 'cmd_vel_y', 'cmd_vel_z'],
        'residual_stats': {
            'mean': residuals.mean(axis=0),
            'std': residuals.std(axis=0),
            'max': np.abs(residuals).max(axis=0)
        },
        'feature_ranges': {
            'velocity_min': features[:, :3].min(axis=0),
            'velocity_max': features[:, :3].max(axis=0),
            'cmd_velocity_min': features[:, 3:].min(axis=0),
            'cmd_velocity_max': features[:, 3:].max(axis=0)
        },
        'n_samples': len(residuals),
        'coordinate_frame': 'ROS (Y=LEFT, post-flip)',
        'notes': 'Trained on Gazebo data with Y-axis flip applied'
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✓ Model saved: {output_file}")
    print(f"  Samples used: {model_data['n_samples']}")
    print(f"  Coordinate frame: {model_data['coordinate_frame']}")
    
    return output_file

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 fit_dynamics_residual.py <residual_data.pkl> "
              "[--tau <value>]")
        print("\nExample:")
        print("  python3 fit_dynamics_residual.py residual_data_20241123_143022.pkl")
        print("  python3 fit_dynamics_residual.py residual_data.pkl --tau 0.15")
        print("\nOptions:")
        print("  --tau: Velocity time constant for simulator (default: 0.1)")
        return
    
    data_file = sys.argv[1]
    
    # Parse optional parameters
    tau = 0.1  # Default time constant
    for i, arg in enumerate(sys.argv[2:], start=2):
        if arg == '--tau' and i + 1 < len(sys.argv):
            tau = float(sys.argv[i + 1])
    
    # Simulator parameters
    # ADJUST THESE TO MATCH YOUR ImprovedCopterSimulator!
    sim_params = {
        'velocity_time_constant': tau,
        # Add other parameters as needed
    }
    
    print("="*60)
    print("DYNAMICS RESIDUAL MODEL FITTING")
    print("="*60)
    print(f"\nInput file: {data_file}")
    print(f"Simulator tau: {tau}")
    
    # Load data
    data = load_data(data_file)
    
    # Compute residuals
    residuals, features, valid_indices = compute_residuals(data, sim_params)
    
    # Analyze
    analyze_residuals(residuals, features, data, valid_indices)
    
    # Fit KNN model
    knn_model, predictions = fit_knn_model(features, residuals, n_neighbors=5)
    
    # Visualize
    visualize_residuals(residuals, predictions, features)
    
    # Save model
    model_file = save_model(knn_model, residuals, features)
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print(f"\n1. Use this model for fine-tuning:")
    print(f"   model_file = '{model_file}'")
    print(f"\n2. Update your simulator to use residuals:")
    print(f"   env = ResidualAugmentedSimulator(")
    print(f"       base_env_config={{...}},")
    print(f"       residual_model_path='{model_file}'")
    print(f"   )")
    print(f"\n3. Fine-tune for 20M steps with augmented sim")
    print("="*60)

if __name__ == '__main__':
    main()
