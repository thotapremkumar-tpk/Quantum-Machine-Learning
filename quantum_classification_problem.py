#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pennylane')

import os
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# In[ ]:


# Fix the random seed so restarts/results are reproducible across runs.
np.random.seed(42)

n_qubits = 4
# n_qubits = 4 because we have exactly 4 input features (4 movie ratings) and
# we're using ONE qubit per feature (angle encoding — see CELL 3).

n_layers = 2
# n_layers = 2 means the variational ansatz (the "trainable" part of the
# circuit) repeats its rotate-then-entangle block twice. More layers = more
# expressive circuit, but also more parameters to train and slower training

dev = qml.device("default.qubit", wires=n_qubits)


# Dataset - updated as large dataset in programmatically generated
# 

# In[ ]:


def generate_dataset(n_scifi=30, n_romance=30, seed=7):
    """
    Generate a larger synthetic movie-rating dataset :
      - SciFi fan  : Movie1, Movie2 ~ high (4-5); Movie3, Movie4 ~ low (1-2)
      - Romance fan: Movie1, Movie2 ~ low (1-2);  Movie3, Movie4 ~ high (4-5)

    Returns shuffled (X_raw, y) arrays, X_raw with raw 1-5 ratings.
    """
    rng = np.random.default_rng(seed)
    rows, labels = [], []

    for _ in range(n_scifi):
        m1, m2 = rng.integers(4, 6), rng.integers(4, 6)   # 4 or 5
        m3, m4 = rng.integers(1, 3), rng.integers(1, 3)   # 1 or 2
        rows.append([m1, m2, m3, m4])
        labels.append(0)  # SciFi

    for _ in range(n_romance):
        m1, m2 = rng.integers(1, 3), rng.integers(1, 3)   # 1 or 2
        m3, m4 = rng.integers(4, 6), rng.integers(4, 6)   # 4 or 5
        rows.append([m1, m2, m3, m4])
        labels.append(1)  # Romance

    rows = np.array(rows, dtype=float)
    labels = np.array(labels, dtype=float)

    # Shuffle so SciFi/Romance rows aren't grouped in order (more realistic,
    # and prevents any accidental ordering bias during training).
    idx = rng.permutation(len(rows))
    return rows[idx], labels[idx]


# Generate a larger, balanced dataset: 30 SciFi + 30 Romance = 60 users
# (vs. the original 11 users, 10 SciFi + 1 Romance). Change n_scifi /
# n_romance here to resize the dataset further.
X_raw, Y_data = generate_dataset(n_scifi=30, n_romance=30, seed=7)
X_raw = np.array(X_raw, requires_grad=False)

# Print the generated raw data table so it's always visible in the notebook
# output, not just hidden inside generate_dataset().
print("Generated dataset (User ID | Movie1-4 ratings | Preferred Category):")
print("-" * 70)
print(f"{'ID':>3} | {'M1':>3} {'M2':>3} {'M3':>3} {'M4':>3} | {'Category':>8}")
print("-" * 70)
for i, (row, label) in enumerate(zip(X_raw, Y_data)):
    cat = "Romance" if label == 1 else "SciFi"
    print(f"{i:3d} | {int(row[0]):3d} {int(row[1]):3d} {int(row[2]):3d} {int(row[3]):3d} | {cat:>8}")
print("-" * 70)
print(f"Total users: {len(Y_data)}  |  SciFi: {int(np.sum(Y_data==0))}  |  Romance: {int(np.sum(Y_data==1))}\n")

# ANGLE ENCODING: rotation gates take an angle, not a raw rating, so we
# linearly rescale ratings (1-5) to angles [0, pi]:
#   rating 1 -> angle 0   (qubit stays near |0>)
#   rating 5 -> angle pi  (qubit rotates near |1>)
X_data = (X_raw - 1) / 4 * np.pi


# Quantum Circuit ( Encoding + Trainable + Measurement)

# In[ ]:


@qml.qnode(dev)
def circuit(theta, x):
    for l in range(n_layers):
        # --- 1) Encode this user's ratings into rotation angles ---
        for i in range(n_qubits):
            qml.RY(x[i], wires=i)

        # --- 2) Trainable rotations (the "learned weights") ---
        for i in range(n_qubits):
            qml.RY(theta[l, i, 0], wires=i)
            qml.RZ(theta[l, i, 1], wires=i)

        # --- 3) Entangle all 4 qubits in a ring ---
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])

    # Measure the expectation value of Pauli-Z on qubit 0. This returns a
    # number in [-1, +1]: -1 means qubit 0 is very likely |1>, +1 means very
    # likely |0>. We turn this into a class probability in CELL 4.

    return qml.expval(qml.PauliZ(0))

_placeholder_theta = np.zeros((n_layers, n_qubits, 2), requires_grad=False)
print("Circuit structure (qml.draw), illustrated with untrained parameters:")
print(qml.draw(circuit)(_placeholder_theta, X_data[0]))
print()


# Turn the circuit's raw output into a probability, define the loss

# In[ ]:


def forward(theta, x):
    z = circuit(theta, x)
    p = (1.0 - z) / 2.0
    return np.clip(p, 1e-9, 1 - 1e-9)

# CLASS IMBALANCE HANDLING:
# The dataset has 10 SciFi vs 1 Romance example. Without correction, a model
# that always predicts "SciFi" would already score 10/11 = 91% accuracy while
# learning nothing useful. pos_weight scales up the loss contribution of the
# single Romance example (~10x) so misclassifying it is penalized as heavily
# as misclassifying all 10 SciFi examples combined.

pos_weight = np.sum(Y_data == 0) / np.sum(Y_data == 1)  # class-imbalance weight (~10.0)


# In[ ]:


def bce_loss(theta, X, Y):
    total = 0.0
    for x, y in zip(X, Y):
        p = forward(theta, x)
        w = float(pos_weight) if y == 1 else 1.0
        total += -w * (y * np.log(p) + (1 - y) * np.log(1 - p))
    return total / len(X)


# In[ ]:


def accuracy(theta, X, Y):
    correct = 0
    for x, y in zip(X, Y):
        pred = 1 if forward(theta, x) > 0.5 else 0
        correct += int(pred == y)
    return correct / len(X)


# Training loop (single restart) + multi-restart driver

# In[ ]:


epochs = 60           # number of gradient-descent steps per restart
LOSS_ZERO_THRESHOLD = 1e-2 # loss below this (AND 100% accuracy) = "converged

def train_once(seed):
    rng = np.random.default_rng(seed)

    # Initialize parameters as small random values (0.8 * standard normal)
    # rather than zeros, since all-zero rotations would start the circuit in
    # a symmetric state with vanishing/uninformative gradients.

    theta = np.array(0.8 * rng.standard_normal((n_layers, n_qubits, 2)), requires_grad=True)

    # Adam optimizer: adapts the learning rate per-parameter, generally
    # converges faster and more reliably than plain gradient descent here

    opt = qml.AdamOptimizer(stepsize=0.2)
    loss_history = []
    acc_history = []
    converged_epoch = None
    for epoch in range(1, epochs + 1):
        # One optimization step: PennyLane computes the gradient of bce_loss
        # w.r.t. theta (via the parameter-shift rule under the hood) and
        # updates theta accordingly.
        theta = opt.step(lambda t: bce_loss(t, X_data, Y_data), theta)
        loss = bce_loss(theta, X_data, Y_data)
        acc = accuracy(theta, X_data, Y_data)
        loss_history.append(float(loss))
        acc_history.append(acc)
        if converged_epoch is None and loss < LOSS_ZERO_THRESHOLD and acc == 1.0:
            converged_epoch = epoch
    return theta, loss_history, acc_history, converged_epoch


# In[ ]:


# MULTI-RESTART TRAINING:
# Variational circuits can get stuck in different local minima depending on
# their random initialization. Running several restarts (different seeds) and
# keeping the best one is standard practice to avoid reporting an unlucky run.

print(" Parametric quantum circuit for Movie Preference (SciFi vs Romance)")
print("=" * 70)

best_theta, best_loss_hist, best_acc_hist, best_converged = None, None, None, None
best_final_loss = np.inf

for seed in range(2):
    theta, loss_hist, acc_hist, converged_epoch = train_once(seed)
    final_loss = loss_hist[-1]
    conv_str = str(converged_epoch) if converged_epoch else "not reached"
    print(f"restart {seed} | final loss = {final_loss:.4f} | "
          f"100% accuracy + near-zero loss first reached at epoch: {conv_str}")
    if final_loss < best_final_loss:
        best_final_loss = final_loss
        best_theta = theta
        best_loss_hist = loss_hist
        best_acc_hist = acc_hist
        best_converged = converged_epoch

# Keep only the best-performing restart's parameters for everything that follows.

theta = best_theta
print(f"\nBest restart final loss: {best_final_loss:.4f}")
if best_converged:
    print(f"Best restart reached 100% precision/recall (all 11 predictions correct) "
          f"and near-zero loss at epoch {best_converged} (out of {epochs}).")
else:
    print("Best restart did not reach the zero-loss threshold within the epoch budget.")


# Final truth table — inspect exactly what the model predicts per user
# 

# In[ ]:


print("\nFinal predictions:")
print("-" * 70)
print(f"{'UserID':>6} | {'target':>8} | {'p(Romance)':>11} | {'predicted':>9}")
print("-" * 70)
correct = 0
for i, (x, y) in enumerate(zip(X_data, Y_data)):
    p = forward(theta, x)
    pred = 1 if p > 0.5 else 0
    correct += int(pred == y)
    true_name = "Romance" if y == 1 else "SciFi"
    pred_name = "Romance" if pred == 1 else "SciFi"
    print(f"{i:6d} | {true_name:>8} | {p:>11.4f} | {pred_name:>9}")
print("-" * 70)
print(f"Accuracy: {correct}/{len(Y_data)}")


# Plot — loss (log scale) & accuracy vs epoch, for the best restart
# 

# In[ ]:


# Log scale on the loss axis makes it easy to see relative improvement even
# once the loss gets small (a drop from 0.5 -> 0.2 looks as visually
# significant as 0.05 -> 0.02, which a linear axis would compress/hide).

fig, ax1 = plt.subplots(figsize=(8, 5))
epochs_range = range(1, epochs + 1)

ax1.plot(epochs_range, best_loss_hist, color="#d62728", linewidth=2, label="BCE loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Binary cross-entropy loss", color="#d62728")
ax1.tick_params(axis="y", labelcolor="#d62728")
ax1.set_yscale("log")

ax2 = ax1.twinx()         # second y-axis sharing the same x-axis, for accuracy
ax2.plot(epochs_range, best_acc_hist, color="#1f77b4", linewidth=2, linestyle="--", label="Accuracy")
ax2.set_ylabel("Accuracy on 11 users", color="#1f77b4")
ax2.tick_params(axis="y", labelcolor="#1f77b4")
ax2.set_ylim(-0.05, 1.05)

# Mark the convergence point, if the strict threshold was ever reached.

if best_converged:
    ax1.axvline(best_converged, color="gray", linestyle=":", linewidth=1.5)
    ax1.text(best_converged + 1, best_loss_hist[0] * 0.5,
              f"converged @ epoch {best_converged}", fontsize=9, color="gray")

plt.title("Movie Preference VQC training: loss and accuracy vs epoch (best restart)")
fig.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/loss_curve.png", dpi=150)
plt.show()


# In[ ]:


# We draw the circuit "for User 0" purely to have concrete numeric encoding
# angles to display — the CIRCUIT STRUCTURE (gates/entanglement) is identical
# for every user; only the encoding angles (x) change per user, while theta
# (the trained weights, already fixed after training) stays the same.

fig2, ax = qml.draw_mpl(circuit, decimals=2, style="pennylane")(theta, X_data[0])
fig2.suptitle("Trained Movie Preference variational circuit (shown for User 0)")
fig2.savefig(f"{OUTPUT_DIR}/circuit_diagram.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nSaved loss curve to {OUTPUT_DIR}/loss_curve.png")
print(f"Saved circuit diagram to {OUTPUT_DIR}/circuit_diagram.png")

