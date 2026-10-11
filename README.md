# Bloomwell — Mental Wellness Prediction Using Machine Learning

A small Flask website mini-project with a calm pastel UI and a scikit-learn Random Forest classifier.

## Features
- Responsive home page and wellness questionnaire
- Flask backend for input validation and prediction
- Random Forest model trained by `train_model.py`
- Result page with general wellness suggestions
- Does not save assessment responses to a user profile

## Important limitation
This is a learning/demo project. The model is trained on **simulated data with illustrative labels**, not clinical data. It is not clinically validated and must not be used for diagnosis, treatment, or important health decisions. Do not present the demo accuracy as real-world mental-health accuracy.

## Requirements
- Python 3.10 or newer recommended
- Internet is optional for the app itself; Google Fonts are loaded from the web when available.

## Setup (Windows)
1. Extract the ZIP file.
2. Open Command Prompt or the VS Code terminal in the project folder.
3. Create a virtual environment:
   ```bash
   py -m venv venv
   ```
4. Activate it:
   ```bash
   venv\Scripts\activate
   ```
5. Install packages:
   ```bash
   python -m pip install -r requirements.txt
   ```
6. Start the website:
   ```bash
   python app.py
   ```
   The first run automatically trains the demo model and creates the `model/` folder.
7. Open the local address printed in the terminal, usually:
   `http://127.0.0.1:5000`

## Setup (macOS/Linux)
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

## Project structure
```text
mental_wellness_website/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── templates/
│   ├── index.html
│   └── result.html
├── static/
│   └── style.css
└── model/                 # generated automatically on first run
    ├── wellness_model.joblib
    └── scaler.joblib
```

## ML workflow
1. Generate simulated survey-style feature data.
2. Split into training and test sets.
3. Standardize the input features with `StandardScaler`.
4. Train a `RandomForestClassifier`.
5. Save the model and scaler with `joblib`.
6. Flask loads both artifacts and predicts a category for the questionnaire responses.

## Input features
- Sleep hours (0–14)
- Stress level (1–5)
- Mood rating (1–5)
- Physical activity days per week (0–7)
- Perceived social support (1–5)
- Recreational screen hours (0–16)

## For a stronger academic version
Replace simulated training data with an appropriate, ethically sourced dataset; document the data source, consent/licensing, preprocessing, class balance, and evaluation. Avoid collecting identifying information and do not claim clinical accuracy without suitable validation.
