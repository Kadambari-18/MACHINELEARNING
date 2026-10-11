"""Creates a small educational demo classifier from simulated survey data.

This is NOT clinical data and the model is NOT clinically validated.
Replace this training pipeline with an ethically sourced, appropriate dataset
before making research or real-world claims.
"""
import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train():
    rng = np.random.default_rng(42)
    n = 1800
    sleep = np.clip(rng.normal(7.1, 1.5, n), 2, 12)
    stress = rng.integers(1, 6, n)
    mood = rng.integers(1, 6, n)        # 1=very low, 5=very good
    activity = rng.integers(0, 8, n)    # days/week
    support = rng.integers(1, 6, n)     # 1=low, 5=high
    screen = np.clip(rng.normal(5.5, 2.5, n), 0, 16)

    # Simulated labels based on transparent, illustrative rules.
    score = (
        (stress - 1) * 1.15
        + (5 - mood) * 1.15
        + np.maximum(0, 6.5 - sleep) * 0.75
        + np.maximum(0, screen - 6) * 0.25
        + np.maximum(0, 2 - activity) * 0.45
        + (5 - support) * 0.45
    )
    labels = np.where(score < 5.4, 0, np.where(score < 9.0, 1, 2))

    X = np.column_stack([sleep, stress, mood, activity, support, screen])
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=140, random_state=42, max_depth=8)
    model.fit(X_train_scaled, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test_scaled))

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/wellness_model.joblib")
    joblib.dump(scaler, "model/scaler.joblib")
    print(f"Demo model trained. Holdout accuracy on simulated labels: {accuracy:.3f}")
    print("Reminder: this score only measures agreement with simulated labels, not real-world validity.")

if __name__ == "__main__":
    train()
