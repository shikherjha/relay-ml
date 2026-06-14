"""Train a logistic regression model for wish-score on synthetic data.

Generates realistic labeled data based on the expected signal relationships:
- Recent wish + many purchases + high affinity + fit profile → high intent (label 1)
- Old wish + few purchases + low affinity + no profile → low intent (label 0)

Saves the trained model to models/wish_logreg_v1.pkl
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Reproducibility
np.random.seed(42)

N_SAMPLES = 2000


def generate_synthetic_data(n: int = N_SAMPLES):
    """Generate synthetic wish-score training data.

    Labels based on realistic buyer-intent heuristics:
    - score = f(recency, purchases, affinity, fit_profile) + noise
    - Threshold at 0.5 to create binary labels
    """
    # Features
    wish_age_days = np.random.uniform(0, 30, n)
    user_purchase_count = np.random.randint(0, 25, n)
    category_affinity = np.random.uniform(0, 1, n)
    has_fit_profile = np.random.choice([0, 1], n, p=[0.5, 0.5])

    # Ground truth intent score (continuous) — calibrated for ~60/40 split
    intent = (
        -0.5
        - 0.04 * wish_age_days  # older = less intent
        + 0.08 * np.minimum(user_purchase_count, 15)  # more purchases = more intent
        + 1.5 * category_affinity  # high affinity = intent
        + 0.7 * has_fit_profile  # fit profile = serious
        + np.random.normal(0, 0.6, n)  # noise
    )

    # Sigmoid to get probability
    prob = 1 / (1 + np.exp(-intent))
    labels = (prob >= 0.5).astype(int)

    X = np.column_stack([wish_age_days, user_purchase_count, category_affinity, has_fit_profile])
    return X, labels


def main():
    print("Generating synthetic training data...")
    X, y = generate_synthetic_data()

    print(f"  Samples: {len(X)}")
    print(f"  Positive (high intent): {y.sum()} ({y.mean():.1%})")
    print(f"  Negative (low intent): {(1-y).sum()} ({(1-y).mean():.1%})")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train
    print("\nTraining logistic regression...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  Test accuracy: {accuracy:.4f}")
    print(f"\n  Coefficients:")
    feature_names = ["wish_age_days", "user_purchase_count", "category_affinity", "has_fit_profile"]
    for name, coef in zip(feature_names, model.coef_[0]):
        print(f"    {name}: {coef:.4f}")
    print(f"    intercept: {model.intercept_[0]:.4f}")

    # Monotonicity check
    print("\n  Monotonicity check:")
    base = np.array([[5, 5, 0.5, 0]])
    newer = np.array([[1, 5, 0.5, 0]])
    more_purch = np.array([[5, 15, 0.5, 0]])
    high_aff = np.array([[5, 5, 0.9, 0]])
    with_fit = np.array([[5, 5, 0.5, 1]])

    print(f"    base (5d, 5p, 0.5a, no-fit): {model.predict_proba(base)[0][1]:.4f}")
    print(f"    newer wish (1d):             {model.predict_proba(newer)[0][1]:.4f}")
    print(f"    more purchases (15):         {model.predict_proba(more_purch)[0][1]:.4f}")
    print(f"    high affinity (0.9):         {model.predict_proba(high_aff)[0][1]:.4f}")
    print(f"    with fit profile:            {model.predict_proba(with_fit)[0][1]:.4f}")

    # Save model
    output_path = Path("models/wish_logreg_v1.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n  Model saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size} bytes")

    # Also save metadata
    meta = {
        "model": "logreg_v1",
        "features": feature_names,
        "coefficients": model.coef_[0].tolist(),
        "intercept": model.intercept_[0],
        "accuracy": accuracy,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    meta_path = Path("models/wish_logreg_v1.metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
