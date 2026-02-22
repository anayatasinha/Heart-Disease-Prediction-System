from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("heart_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [
        float(request.form["age"]),
        float(request.form["sex"]),
        float(request.form["cp"]),
        float(request.form["trestbps"]),
        float(request.form["chol"]),
        float(request.form["fbs"]),
        float(request.form["restecg"]),
        float(request.form["thalach"]),
        float(request.form["exang"]),
        float(request.form["oldpeak"]),
        float(request.form["slope"]),
        float(request.form["ca"]),
        float(request.form["thal"])
    ]

    data = pd.DataFrame([features], columns=[
        "age","sex","cp","trestbps","chol","fbs",
        "restecg","thalach","exang","oldpeak",
        "slope","ca","thal"
    ])

    # prediction = model.predict(data)[0]
    proba = model.predict_proba(data)[0]
    classes = model.classes_

    heart_class = 0  
    probability = proba[list(classes).index(heart_class)]

    if probability >= 0.75:
        result = "🚨 High Risk – Please consult a cardiologist"
    elif probability >= 0.45:
        result = "⚠️ Moderate Risk – Medical check advised"
    else:
        result = "✅ Low Risk – No immediate concern"

    return render_template(
        "index.html",
        prediction_text=result,
        probability=f"{probability*100:.2f}%"
    )

if __name__ == "__main__":
    app.run(debug=True)