import os
import torch
import numpy as np
from flybrain import FlyBrain
from sklearn.linear_model import LogisticRegression
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

# STEP 5: Run Data Collection Loop
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

# STEP 6: Train, Evaluate, and Track Performance
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

# STEP 7: Visualizations
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


from sklearn.linear_model import SGDClassifier
# Initialize an online-learning classifier instead of static LogisticRegression
# loss='log_loss' gives us probabilistic outputs similar to Logistic Regression
decoder = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)

# Define our scale labels for partial fitting
all_scale_names = list(music_scales.keys())

print()
print("--- Training Fly Brain with Iterative Learning (Epochs) ---")
epochs = 5

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1} of {epochs}")
    correct_count = 0
    total_count = 0
    
    for scale_name, pitch_vec in music_scales.items():
        for trial in range(5):  # 5 trials per epoch per scale
            # Add noise
            noisy_input = np.array(pitch_vec, dtype=float) + np.random.normal(0, 0.05, 12)
            binary_noisy = (noisy_input > 0.5).astype(int)
            
            # Keep the SAME 'brain' instance alive across trials and epochs
            brain_state = simulate_musical_input(brain, pitch_bins, binary_noisy, steps=20)
            
            X_sample = brain_state.reshape(1, -1)
            y_sample = np.array([scale_name])
            
            # If the decoder has already seen data, test its guess BEFORE it learns from this trial
            if epoch > 0 or total_count > 0:
                pred = decoder.predict(X_sample)
                if pred[0] == scale_name:
                    correct_count += 1
            
            total_count += 1
            
            # Incrementally update (learn) the decoder weights using this single experience
            decoder.partial_fit(X_sample, y_sample, classes=all_scale_names)

    if epoch > 0:
        epoch_accuracy = (correct_count / total_count) * 100
        print(f"  -> Epoch {epoch + 1} Accuracy: {epoch_accuracy:.2f}%")

print("\nLearning Complete")