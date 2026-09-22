#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib
matplotlib.use('Agg')  # headless backend — no display window needed
import matplotlib.pyplot as plt

OUTPUT_DIR = "outputs/classical"
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(42)

n_estimators = 200      # number of boosting rounds ("epochs" for this model)
# n_estimators = 200 to match the quantum script's epoch budget, so the two
# models get a comparable amount of training.

LOSS_ZERO_THRESHOLD = 1e-2  # loss below this (AND 100% accuracy) = "converged"


# ---------------------------------------------------------------------------
# Data loading & preprocessing (identical to quantum_classification_problem.py
# so the two models are trained/evaluated on exactly the same data)
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


def setup_class_weights(Y_data):
    """Compute pos_weight to handle class imbalance between SciFi and Romance."""
    n_scifi = int(np.sum(Y_data == 0))
    n_romance = int(np.sum(Y_data == 1))
    pos_weight = n_scifi / n_romance if n_romance > 0 else 1.0
    print(f"Using pos_weight = {pos_weight:.4f} for Romance examples")
    return pos_weight


# ---------------------------------------------------------------------------
# Classical model: Gradient Boosted Trees
#
# Chosen as the classical counterpart to the variational quantum circuit
# because, like the VQC, it trains in discrete stages (boosting rounds ~
# epochs), so we get a directly comparable per-stage loss/accuracy curve.
# Gradient-boosted trees are also consistently one of the strongest classical
# baselines on small, tabular datasets like this one.
# ---------------------------------------------------------------------------

def bce_loss_from_proba(p_romance, Y, pos_weight):
    """Binary cross-entropy loss, given already-predicted probabilities."""
    p = np.clip(p_romance, 1e-9, 1 - 1e-9)
    w = np.where(Y == 1, float(pos_weight), 1.0)
    return float(np.mean(-w * (Y * np.log(p) + (1 - Y) * np.log(1 - p))))


def train_once(seed, X_data, Y_data, pos_weight, n_estimators=60):
    sample_weight = np.where(Y_data == 1, float(pos_weight), 1.0)

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=0.1,
        max_depth=3,
        random_state=seed,
    )
    model.fit(X_data, Y_data, sample_weight=sample_weight)

    loss_history = []
    acc_history = []
    converged_epoch = None
    for stage, proba in enumerate(model.staged_predict_proba(X_data), start=1):
        p_romance = proba[:, 1]
        loss = bce_loss_from_proba(p_romance, Y_data, pos_weight)
        acc = float(np.mean((p_romance > 0.5).astype(float) == Y_data))
        loss_history.append(loss)
        acc_history.append(acc)
        if converged_epoch is None and loss < LOSS_ZERO_THRESHOLD and acc == 1.0:
            converged_epoch = stage

    return model, loss_history, acc_history, converged_epoch


# ---------------------------------------------------------------------------
# Modular helpers called by main() (mirrors quantum_classification_problem.py)
# ---------------------------------------------------------------------------

def run_multi_restart_training(X_data, Y_data, pos_weight, n_restarts=4):
    """Run training from multiple random seeds; return the best result."""
    print(" Gradient Boosted Trees for Movie Preference (SciFi vs Romance)")
    print("=" * 70)

    best_model, best_loss_hist, best_acc_hist, best_converged = None, None, None, None
    best_final_loss = np.inf

    for seed in range(n_restarts):
        model, loss_hist, acc_hist, converged_epoch = train_once(
            seed, X_data, Y_data, pos_weight, n_estimators
        )
        final_loss = loss_hist[-1]
        conv_str = str(converged_epoch) if converged_epoch else "not reached"
        print(
            f"restart {seed} | final loss = {final_loss:.4f} | "
            f"100% accuracy + near-zero loss first reached at epoch: {conv_str}"
        )
        if final_loss < best_final_loss:
            best_final_loss = final_loss
            best_model = model
            best_loss_hist = loss_hist
            best_acc_hist = acc_hist
            best_converged = converged_epoch

    print(f"\nBest restart final loss: {best_final_loss:.4f}")
    if best_converged:
        print(f"Best restart reached 100% accuracy and near-zero loss at epoch {best_converged}.")
    else:
        print("Best restart did not reach the zero-loss threshold within the epoch budget.")

    return best_model, best_loss_hist, best_acc_hist, best_converged


def print_final_predictions(model, X_data, Y_data):
    """Print a formatted table of per-user predictions vs ground-truth labels."""
    print("\nFinal predictions:")
    print("-" * 70)
    print(f"{'UserID':>6} | {'target':>8} | {'p(Romance)':>11} | {'predicted':>9}")
    print("-" * 70)
    proba = model.predict_proba(X_data)[:, 1]
    correct = 0
    for i, (p, y) in enumerate(zip(proba, Y_data)):
        pred = 1 if p > 0.5 else 0
        correct += int(pred == y)
        true_name = "Romance" if y == 1 else "SciFi"
        pred_name = "Romance" if pred == 1 else "SciFi"
        print(f"{i:6d} | {true_name:>8} | {p:>11.4f} | {pred_name:>9}")
    print("-" * 70)
    print(f"Accuracy: {correct}/{len(Y_data)}")


def plot_training_curves(loss_hist, acc_hist, converged_epoch):
    """Plot BCE loss and accuracy vs boosting round and save the figure."""
    fig, ax1 = plt.subplots(figsize=(8, 5))
    epochs_range = range(1, len(loss_hist) + 1)

    ax1.plot(epochs_range, loss_hist, color="#d62728", linewidth=2, label="BCE loss")
    ax1.set_xlabel("Boosting round")
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
            f"converged @ round {converged_epoch}",
            fontsize=9,
            color="gray",
        )

    plt.title("Movie Preference Gradient Boosting training: loss and accuracy vs round (best restart)")
    fig.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/loss_curve.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved loss curve to {OUTPUT_DIR}/loss_curve.png")


def save_feature_importance(model):
    """Plot and save the trained model's feature importances (classical analogue
    of the quantum script's circuit diagram — shows what the model learned)."""
    feature_names = [
        "Movie 1 (Sci-Fi)",
        "Movie 2 (Sci-Fi)",
        "Movie 3 (Romance)",
        "Movie 4 (Romance)",
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(feature_names, model.feature_importances_, color="#2ca02c")
    ax.set_ylabel("Feature importance")
    ax.set_title("Trained Gradient Boosting feature importances")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved feature importance plot to {OUTPUT_DIR}/feature_importance.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    X_data, Y_data = load_dataset()

    pos_weight = setup_class_weights(Y_data)

    best_model, best_loss_hist, best_acc_hist, best_converged = run_multi_restart_training(
        X_data, Y_data, pos_weight
    )

    print_final_predictions(best_model, X_data, Y_data)
    plot_training_curves(best_loss_hist, best_acc_hist, best_converged)
    save_feature_importance(best_model)


if __name__ == "__main__":
    main()
