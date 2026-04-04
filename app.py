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

        disease, confidence = predict_image(filepath)

        return render_template(
            "upload.html",
            prediction=disease,
            confidence=round(confidence,2),
            image_path=filepath
        )

if __name__ == "__main__":
    app.run(debug=True)