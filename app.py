from flask import Flask, render_template, request
import os
from predict import predict_image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["file"]

    if file:
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        TREATMENTS = {
            "Bacterial Leaf Blight": "Use copper-based fungicides immediately. Drain the field to reduce humidity, and avoid excessive nitrogen fertilizers which can worsen the disease.",
            "Brown Spot": "Apply appropriate fungicides like Propiconazole. Ensure proper soil fertility with balanced potassium and phosphorus levels.",
            "Rice Hispa": "Manually remove and destroy infested leaves. For severe infestations, consult a local expert for recommended contact insecticides.",
            "Healthy Rice Leaf": "Your crop looks excellent! Continue standard fertilizing and watering routines to maintain good crop health."
        }

        disease, confidence = predict_image(filepath)
        treatment_info = TREATMENTS.get(disease, "Consult local agricultural experts for best practices.")

        return render_template(
            "upload.html",
            prediction=disease,
            confidence=round(confidence,2),
            image_path=filepath,
            treatment=treatment_info
        )

if __name__ == "__main__":
    app.run(debug=True)