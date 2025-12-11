import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# --- 1. Define Paths and Column Names ---

# Directory where your processed data is stored
DATA_DIR = os.path.expanduser("~/mpc_data/processed_drone_data")
TRAIN_FILE = os.path.join(DATA_DIR, 'train_scaled_dataset.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test_scaled_dataset.csv')

# Directory to save the final trained model
MODEL_OUTPUT_DIR = os.path.expanduser("~/drone_student_model")
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODEL_OUTPUT_DIR, 'student_controller.h5')

# Define feature (input) and label (output) columns
FEATURES = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'xt', 'yt', 'zt']
LABELS = ['vmx', 'vmy', 'vmz']

# --- 2. Load the Datasets ---
print(f"Loading training data from: {TRAIN_FILE}")
train_df = pd.read_csv(TRAIN_FILE)
print(f"Loading validation (test) data from: {TEST_FILE}")
test_df = pd.read_csv(TEST_FILE)

# --- 3. Separate Features and Labels ---
X_train_scaled = train_df[FEATURES].values
y_train_scaled = train_df[LABELS].values

X_val_scaled = test_df[FEATURES].values
y_val_scaled = test_df[LABELS].values

print(f"Training data shape: {X_train_scaled.shape}")
print(f"Validation data shape: {X_val_scaled.shape}")


# --- 4. Build the Neural Network Model ---
# (This is the model architecture you provided)
print("\nBuilding model...")
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(9,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3) # Output layer has 3 neurons (vmx, vmy, vmz) with linear activation
])

# --- 5. Compile the Model ---
# We use Mean Squared Error because this is a regression problem.
# The Adam optimizer is a great general-purpose choice.
model.compile(optimizer='adam', 
              loss='mean_squared_error', 
              metrics=['mean_absolute_error'])

model.summary()

# --- 6. Train the Model ---
print("\nStarting model training...")
history = model.fit(
    X_train_scaled,
    y_train_scaled,
    epochs=50,  # Number of passes through the entire dataset
    batch_size=32,
    validation_data=(X_val_scaled, y_val_scaled), # Use the test set for validation
    verbose=1
)

# --- 7. Save the Trained Model ---
model.save(MODEL_SAVE_PATH)
print(f"\nTraining complete! Model saved to {MODEL_SAVE_PATH}")

# --- 8. Visualize Training ---
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plt.show()

print("All done. You can now use the 'student_controller.h5' file in your ROS node.")