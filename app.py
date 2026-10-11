from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)
MODEL_PATH = os.path.join("model", "wellness_model.joblib")
SCALER_PATH = os.path.join("model", "scaler.joblib")

# Train the demo model automatically on first launch if model files are missing.
if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
    from train_model import train
    train()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

FEATURES = [
    "sleep_hours", "stress_level", "mood_level",
    "activity_days", "social_support", "screen_hours"
]

def get_guidance(category):
    common = [
        "Try keeping a regular sleep and wake-up routine.",
        "Take short breaks from screens and make time for an activity you enjoy.",
        "Consider talking with someone you trust about how you have been feeling."
    ]
    if category == "Lower concern":
        return [
            "Keep doing the everyday activities that help you feel balanced.",
            "Protect your sleep routine and make time for movement and connection.",
            "Check in with yourself regularly; wellbeing can change over time."
        ]
    if category == "Moderate concern":
        return [
            "Try a small, realistic routine change, such as a consistent bedtime.",
            "Break demanding tasks into smaller steps and take regular pauses.",
            "If these feelings persist or affect daily life, consider speaking with a qualified professional."
        ]
    return [
        "Reach out to a trusted person and let them know how things have been going.",
        "Consider contacting a qualified mental-health professional or a local support service.",
        "Focus on one manageable next step today; you do not need to handle everything alone."
    ]

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/assess", methods=["POST"])
def assess():
    try:
        sleep = float(request.form["sleep_hours"])
        stress = int(request.form["stress_level"])
        mood = int(request.form["mood_level"])
        activity = int(request.form["activity_days"])
        support = int(request.form["social_support"])
        screen = float(request.form["screen_hours"])
    except (KeyError, ValueError):
        return render_template("index.html", error="Please complete every field with a valid answer.")

    if not (0 <= sleep <= 14 and 1 <= stress <= 5 and 1 <= mood <= 5
            and 0 <= activity <= 7 and 1 <= support <= 5 and 0 <= screen <= 24):
        return render_template("index.html", error="Please check your answers and try again.")

    X = np.array([[sleep, stress, mood, activity, support, screen]], dtype=float)
    prediction = int(model.predict(scaler.transform(X))[0])
    categories = {0: "Lower concern", 1: "Moderate concern", 2: "Higher concern"}
    category = categories[prediction]
    descriptions = {
        "Lower concern": "Your answers suggest fewer of the demo model's wellness risk indicators today.",
        "Moderate concern": "Your answers include some indicators that may be worth paying attention to.",
        "Higher concern": "Your answers include several indicators that may deserve extra care and support."
    }
    return render_template(
        "result.html",
        category=category,
        description=descriptions[category],
        guidance=get_guidance(category),
        values={
            "sleep": sleep, "stress": stress, "mood": mood,
            "activity": activity, "support": support, "screen": screen
        }
    )

if __name__ == "__main__":
    app.run(debug=True)
