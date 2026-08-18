#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import pennylane as qml
from pennylane import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless backend — no display window needed
import matplotlib.pyplot as plt

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(42)

n_qubits = 4
# n_qubits = 4 because there are four movie rating features.

n_layers = 2
# n_layers = 2 means the variational ansatz repeats its rotate-then-entangle
# block twice. This is a reasonable tradeoff between expressivity and speed.

"""0: ──RY(3.14)──RY(0.00)──RZ(0.00)─╭●───────╭X──RY(3.14)──RY(0.00)──RZ(0.00)─╭●───────╭X─┤  <Z>
1: ──RY(2.36)──RY(0.00)──RZ(0.00)─╰X─╭●────│───RY(2.36)──RY(0.00)──RZ(0.00)─╰X─╭●────│──┤     
2: ──RY(0.00)──RY(0.00)──RZ(0.00)────╰X─╭●─│───RY(0.00)──RY(0.00)──RZ(0.00)────╰X─╭●─│──┤     
3: ──RY(0.00)──RY(0.00)──RZ(0.00)───────╰X─╰●──RY(0.00)──RY(0.00)──RZ(0.00)───────╰X─╰●─┤ 
"""
dev = qml.device("default.qubit", wires=n_qubits)


# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------

def load_dataset(path="movielens_sample.csv"):
    df = pd.read_csv(path)
    required_columns = [
        "Movie 1 (Sci-Fi)",
        "Movie 2 (Sci-Fi)",
        "Movie 3 (Romance)",
        "Movie 4 (Romance)",
        "Preferred Category",
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    X_raw = df[[
        "Movie 1 (Sci-Fi)",
        "Movie 2 (Sci-Fi)",
        "Movie 3 (Romance)",
        "Movie 4 (Romance)",
    ]].to_numpy(dtype=float)
    labels = df["Preferred Category"].astype(str).str.strip().str.lower()
    Y_data = np.array([
        0.0 if label == "scifi" else 1.0 if label == "romance" else np.nan
        for label in labels
    ])
    if np.isnan(Y_data).any():
        raise ValueError("Unknown labels found in Preferred Category. Only SciFi and Romance are supported.")
    return X_raw, Y_data


def angle_encode(X_raw):
    if np.nanmin(X_raw) < 0 or np.nanmax(X_raw) > 5:
        raise ValueError("Rating values must be between 0 and 5.")
    return X_raw / 5.0 * np.pi


def print_dataset_summary(X_raw, Y_data):
    print("Loaded dataset (User ID | Movie1-4 ratings | Preferred Category):")
    print("-" * 70)
    print(f"{'ID':>3} | {'M1':>3} {'M2':>3} {'M3':>3} {'M4':>3} | {'Category':>8}")
    print("-" * 70)
    for i, (row, label) in enumerate(zip(X_raw, Y_data)):
        cat = "Romance" if label == 1 else "SciFi"
        print(f"{i:3d} | {int(row[0]):3d} {int(row[1]):3d} {int(row[2]):3d} {int(row[3]):3d} | {cat:>8}")
    print("-" * 70)
    print(f"Total users: {len(Y_data)}  |  SciFi: {int(np.sum(Y_data == 0))}  |  Romance: {int(np.sum(Y_data == 1))}\n")


# ---------------------------------------------------------------------------
# Quantum Circuit ( Encoding + Trainable + Measurement)
# ---------------------------------------------------------------------------

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
    # likely |0>. We turn this into a class probability below.

    return qml.expval(qml.PauliZ(0))


# ---------------------------------------------------------------------------
# Turn the circuit's raw output into a probability, define the loss
# ---------------------------------------------------------------------------

def forward(theta, x):
    z = circuit(theta, x)
    p = (1.0 - z) / 2.0
    return np.clip(p, 1e-9, 1 - 1e-9)


def bce_loss(theta, X, Y, pos_weight):
    """Binary cross-entropy loss.

    BCE converts the raw circuit output into a probability and compares it
    to the true binary label. It penalizes confident wrong predictions more
    strongly than uncertain ones, which is appropriate for binary
    classification.
    """
    total = 0.0
    for x, y in zip(X, Y):
        p = forward(theta, x)
        w = float(pos_weight) if y == 1 else 1.0
        total += -w * (y * np.log(p) + (1 - y) * np.log(1 - p))
    return total / len(X)


def accuracy(theta, X, Y):
    correct = 0
    for x, y in zip(X, Y):
        pred = 1 if forward(theta, x) > 0.5 else 0
        correct += int(pred == y)
    return correct / len(X)


# ---------------------------------------------------------------------------
# Training loop (single restart) + multi-restart driver
# ---------------------------------------------------------------------------

epochs = 60            # number of gradient-descent steps per restart
LOSS_ZERO_THRESHOLD = 1e-2  # loss below this (AND 100% accuracy) = "converged"


def train_once(seed, X_data, Y_data, pos_weight, epochs=60):
    rng = np.random.default_rng(seed)

    # Initialize parameters as small random values (0.8 * standard normal)
    # rather than zeros, since all-zero rotations would start the circuit in
    # a symmetric state with vanishing/uninformative gradients.

    theta = np.array(0.8 * rng.standard_normal((n_layers, n_qubits, 2)), requires_grad=True)

    # Adam optimizer: adapts the learning rate per-parameter, generally
    # converges faster and more reliably than plain gradient descent here.

    opt = qml.AdamOptimizer(stepsize=0.2)
    loss_history = []
    acc_history = []
    converged_epoch = None
    for epoch in range(1, epochs + 1):
        theta = opt.step(lambda t: bce_loss(t, X_data, Y_data, pos_weight), theta)
        loss = bce_loss(theta, X_data, Y_data, pos_weight)
        acc = accuracy(theta, X_data, Y_data)
        loss_history.append(float(loss))
        acc_history.append(acc)
        if converged_epoch is None and loss < LOSS_ZERO_THRESHOLD and acc == 1.0:
            converged_epoch = epoch
    return theta, loss_history, acc_history, converged_epoch


# ---------------------------------------------------------------------------
# Modular helpers called by main()
# ---------------------------------------------------------------------------

def setup_class_weights(Y_data):
    """Compute pos_weight to handle class imbalance between SciFi and Romance."""
    n_scifi = int(np.sum(Y_data == 0))
    n_romance = int(np.sum(Y_data == 1))
    pos_weight = n_scifi / n_romance if n_romance > 0 else 1.0
    print(f"Using pos_weight = {pos_weight:.4f} for Romance examples")
    return pos_weight


def draw_circuit(X_data):
    """Print a human-readable diagram of the untrained circuit."""
    placeholder_theta = np.zeros((n_layers, n_qubits, 2), requires_grad=False)
    print("Circuit structure (qml.draw), illustrated with untrained parameters:")
    print(qml.draw(circuit)(placeholder_theta, X_data[0]))
    print()


def run_multi_restart_training(X_data, Y_data, pos_weight, n_restarts=2):
    """Run training from multiple random seeds; return the best result."""
    print(" Parametric quantum circuit for Movie Preference (SciFi vs Romance)")
    print("=" * 70)

    best_theta, best_loss_hist, best_acc_hist, best_converged = None, None, None, None
    best_final_loss = np.inf

    for seed in range(n_restarts):
        theta, loss_hist, acc_hist, converged_epoch = train_once(
            seed, X_data, Y_data, pos_weight
        )
        final_loss = loss_hist[-1]
        conv_str = str(converged_epoch) if converged_epoch else "not reached"
        print(
            f"restart {seed} | final loss = {final_loss:.4f} | "
            f"100% accuracy + near-zero loss first reached at epoch: {conv_str}"
        )
        if final_loss < best_final_loss:
            best_final_loss = final_loss
            best_theta = theta
            best_loss_hist = loss_hist
            best_acc_hist = acc_hist
            best_converged = converged_epoch

    print(f"\nBest restart final loss: {best_final_loss:.4f}")
    if best_converged:
        print(f"Best restart reached 100% accuracy and near-zero loss at epoch {best_converged}.")
    else:
        print("Best restart did not reach the zero-loss threshold within the epoch budget.")

    return best_theta, best_loss_hist, best_acc_hist, best_converged


def print_final_predictions(theta, X_data, Y_data):
    """Print a formatted table of per-user predictions vs ground-truth labels."""
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


def plot_training_curves(loss_hist, acc_hist, converged_epoch):
    """Plot BCE loss and accuracy vs epoch and save the figure."""
    fig, ax1 = plt.subplots(figsize=(8, 5))
    epochs_range = range(1, len(loss_hist) + 1)

    ax1.plot(epochs_range, loss_hist, color="#d62728", linewidth=2, label="BCE loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Binary cross-entropy loss", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_yscale("log")

    ax2 = ax1.twinx()
    ax2.plot(epochs_range, acc_hist, color="#1f77b4", linewidth=2, linestyle="--", label="Accuracy")
    ax2.set_ylabel("Accuracy", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.set_ylim(-0.05, 1.05)

    if converged_epoch:
        ax1.axvline(converged_epoch, color="gray", linestyle=":", linewidth=1.5)
        ax1.text(
            converged_epoch + 1,
            loss_hist[0] * 0.5,
            f"converged @ epoch {converged_epoch}",
            fontsize=9,
            color="gray",
        )

    plt.title("Movie Preference VQC training: loss and accuracy vs epoch (best restart)")
    fig.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/loss_curve.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved loss curve to {OUTPUT_DIR}/loss_curve.png")


def save_circuit_diagram(theta, X_data):
    """Render and save the trained circuit diagram for the first user."""
    fig2, ax = qml.draw_mpl(circuit, decimals=2, style="pennylane")(theta, X_data[0])
    fig2.suptitle("Trained Movie Preference variational circuit (shown for User 0)")
    fig2.savefig(f"{OUTPUT_DIR}/circuit_diagram.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved circuit diagram to {OUTPUT_DIR}/circuit_diagram.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    X_raw, Y_data = load_dataset()
    # print_dataset_summary(X_raw, Y_data)  # commented out to avoid bloating stdout
    X_data = angle_encode(X_raw)

    pos_weight = setup_class_weights(Y_data)
    draw_circuit(X_data)

    best_theta, best_loss_hist, best_acc_hist, best_converged = run_multi_restart_training(
        X_data, Y_data, pos_weight
    )

    print_final_predictions(best_theta, X_data, Y_data)
    plot_training_curves(best_loss_hist, best_acc_hist, best_converged)
    save_circuit_diagram(best_theta, X_data)


if __name__ == "__main__":
    main()
