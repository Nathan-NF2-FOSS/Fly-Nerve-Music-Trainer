import os
import torch
import numpy as np
from flybrain import FlyBrain
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# STEP 1: Initialize Environment & Brain
os.environ["FLY_DEVICE"] = "cpu"

print("Initializing fly brain simulation (166,700 neurons)")
brain = FlyBrain()

# STEP 2: Map Sensory Neurons to Pitch Bins
sensory_indices = brain.cells(["LC4", "LPLC2"])
pitch_bins = np.array_split(sensory_indices, 12)

# STEP 3: Define the Simulation Loop
def simulate_musical_input(brain, pitch_bins, pitch_vector, steps=30):
    brain.reset()  
    
    recorded_states = []
    for t in range(steps):
        for note_idx, active in enumerate(pitch_vector):
            if active == 1 and len(pitch_bins[note_idx]) > 0:
                brain.stimulate(pitch_bins[note_idx], 0.8)
                
        brain.step()
        
        v = brain.v
        if hasattr(v, 'get'):
            v = v.get()
        state_vector = np.array(v).flatten()[:2000]
        recorded_states.append(state_vector)
        
    return np.mean(recorded_states, axis=0)

# STEP 4: Define College-Scale Dataset
music_scales = {
    "Major (Ionian)": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    "Harmonic Minor": [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1],
    "Melodic Minor (Ascending)": [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    "Locrian #2": [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
    "Super Locrian (Altered)": [1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
    "Lydian Augmented": [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1],
    "Double Harmonic Major": [1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1]
}

# STEP 5: Run Data Collection Loop (Batch / Static)
X_data = []
y_labels = []

print("Running simulation sweeps across scale dataset...")
for scale_name, pitch_vec in music_scales.items():
    for trial in range(10):
        noisy_input = np.array(pitch_vec, dtype=float) + np.random.normal(0, 0.05, 12)
        binary_noisy = (noisy_input > 0.5).astype(int)
        
        brain_state = simulate_musical_input(brain, pitch_bins, binary_noisy, steps=30)
        X_data.append(brain_state)
        y_labels.append(scale_name)

X_data = np.array(X_data)
y_labels = np.array(y_labels)

# STEP 6: Train, Evaluate, and Track Static Performance
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_labels, test_size=0.3, random_state=42
)

decoder = LogisticRegression(max_iter=1000)
decoder.fit(X_train, y_train)

y_pred = decoder.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nPipeline Complete!")
print(f"Overall Decoder Accuracy: {accuracy * 100:.2f}%\n")

print("--- Per-Scale Performance Tracker ---")
scale_tracker = {}
for true_label, pred_label in zip(y_test, y_pred):
    if true_label not in scale_tracker:
        scale_tracker[true_label] = {"correct": 0, "total": 0}
    
    scale_tracker[true_label]["total"] += 1
    if true_label == pred_label:
        scale_tracker[true_label]["correct"] += 1

for scale, stats in scale_tracker.items():
    correct = stats["correct"]
    total = stats["total"]
    pct = (correct / total) * 100 if total > 0 else 0
    print(f"  • {scale:<25}: {correct}/{total} correct ({pct:.1f}%)")

# STEP 7: Visualizations (Static State & Initial Confusion Matrix)
sample_states = []
scale_names_list = list(music_scales.keys())

for scale_name in scale_names_list:
    pitch_vec = music_scales[scale_name]
    brain_state = simulate_musical_input(brain, pitch_bins, pitch_vec, steps=30)
    sample_states.append(brain_state)

plt.figure(figsize=(10, 6))
sns.heatmap(sample_states, cmap="viridis", yticklabels=scale_names_list)
plt.title("Fly Brain Readout Activations per Music Scale")
plt.xlabel("Neuron Voltage Feature Sample (First 2,000)")
plt.ylabel("College Scale")
plt.tight_layout()
plt.show()

cm = confusion_matrix(y_test, y_pred, labels=scale_names_list)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", 
            xticklabels=scale_names_list, 
            yticklabels=scale_names_list)
plt.title(f"Decoder Confusion Matrix (Overall Accuracy: {accuracy * 100:.1f}%)")
plt.xlabel("Predicted Scale")
plt.ylabel("Actual Scale")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


from sklearn.preprocessing import StandardScaler

# STEP 8: Iterative Learning with Baseline (Epoch 0) Evaluation
print("\nPreparing fixed evaluation dataset...")
X_eval_raw = []
y_eval = []
for scale_name, pitch_vec in music_scales.items():
    for _ in range(10):
        noisy_input = np.array(pitch_vec, dtype=float) + np.random.normal(0, 0.05, 12)
        binary_noisy = (noisy_input > 0.5).astype(int)
        brain_state = simulate_musical_input(brain, pitch_bins, binary_noisy, steps=20)
        X_eval_raw.append(brain_state)
        y_eval.append(scale_name)

X_eval_raw = np.array(X_eval_raw)
y_eval = np.array(y_eval)

# Initialize Scaler and fit on evaluation data
scaler = StandardScaler()
X_eval = scaler.fit_transform(X_eval_raw)

# SGDClassifier starts with random weights (untrained baseline)
decoder = SGDClassifier(
    loss='log_loss', 
    max_iter=1000, 
    learning_rate='constant', 
    eta0=0.001, 
    alpha=0.0001,
    random_state=42
)
all_scale_names = list(music_scales.keys())

epochs = 10
accuracy_history = []

# --- 1. EVALUATE UNTRAINED BASELINE (Epoch 0) ---
print("\nEvaluating Untrained Baseline (Epoch 0)...")
# Force classes initialization without fitting weights yet
decoder.partial_fit(X_eval[0].reshape(1, -1), np.array([y_eval[0]]), classes=all_scale_names)
# Reset weights back to completely random/zero state for a true clean slate
decoder.coef_ = np.zeros_like(decoder.coef_)
decoder.intercept_ = np.zeros_like(decoder.intercept_)

y_pred_baseline = decoder.predict(X_eval)
baseline_accuracy = accuracy_score(y_eval, y_pred_baseline) * 100
accuracy_history.append(baseline_accuracy)

cm_baseline = confusion_matrix(y_eval, y_pred_baseline, labels=all_scale_names)
plt.figure(figsize=(7, 5))
sns.heatmap(cm_baseline, annot=True, fmt="d", cmap="Purples",
            xticklabels=all_scale_names, yticklabels=all_scale_names)
plt.title(f"Epoch 0 (Untrained Baseline)\nAccuracy: {baseline_accuracy:.1f}%")
plt.xlabel("Predicted Scale")
plt.ylabel("Actual Scale")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# --- 2. TRAINING EPOCHS ---
print("\nTraining Fly Brain Across Epochs:")
for epoch in range(epochs):
    if epoch < 4:
        phase = "GROWTH (Learning phase)"
        noise_level = 0.02 + (epoch * 0.03)
    elif epoch < 7:
        phase = "PEAK (Optimal cognitive state)"
        noise_level = 0.12
    else:
        phase = "DECAY (Cognitive overload / heavy noise)"
        noise_level = 0.25 + ((epoch - 6) * 0.15)
        
    print(f"\n--- Epoch {epoch + 1} of {epochs} [{phase}] ---")
    
    X_epoch_train = []
    y_epoch_train = []
    
    for scale_name, pitch_vec in music_scales.items():
        for trial in range(3):
            noisy_input = np.array(pitch_vec, dtype=float) + np.random.normal(0, noise_level, 12)
            binary_noisy = (noisy_input > 0.5).astype(int)
            
            brain_state = simulate_musical_input(brain, pitch_bins, binary_noisy, steps=20)
            X_epoch_train.append(brain_state)
            y_epoch_train.append(scale_name)
            
    X_epoch_train = scaler.transform(np.array(X_epoch_train))
    y_epoch_train = np.array(y_epoch_train)
    
    # Train sequentially on this epoch's data
    for x_sample, y_sample in zip(X_epoch_train, y_epoch_train):
        decoder.partial_fit(x_sample.reshape(1, -1), np.array([y_sample]), classes=all_scale_names)
            
    # Evaluate performance after this epoch's training
    y_pred_epoch = decoder.predict(X_eval)
    epoch_accuracy = accuracy_score(y_eval, y_pred_epoch) * 100
    accuracy_history.append(epoch_accuracy)
    
    print(f"  -> Noise Level: {noise_level:.2f} | Test Accuracy: {epoch_accuracy:.2f}%")
    
    cm_epoch = confusion_matrix(y_eval, y_pred_epoch, labels=all_scale_names)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm_epoch, annot=True, fmt="d", cmap="Purples",
                xticklabels=all_scale_names, yticklabels=all_scale_names)
    plt.title(f"Epoch {epoch + 1} ({phase})\nAccuracy: {epoch_accuracy:.1f}% | Noise: {noise_level:.2f}")
    plt.xlabel("Predicted Scale")
    plt.ylabel("Actual Scale")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# Final Learning Curve Summary
plt.figure(figsize=(8, 4))
plt.plot(range(0, epochs + 1), accuracy_history, marker='o', color='b', linestyle='-', linewidth=2)
plt.title("Fly Brain Cognitive Learning & Decay Curve (Epoch 0 to 10)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.ylim(0, 105)
plt.tight_layout()
plt.show()

print("\nSimulation Complete!")
