# 🌾 PaddyGuard AI

**PaddyGuard AI** is an intelligent web application designed for the fast and accurate detection of paddy (rice) leaf diseases. It is built using Python, deep learning (transfer learning with MobileNetV2), and a responsive Flask web backend.

## ✨ Features
*   **Highly Accurate AI Model:** Instantly detects key paddy diseases/pests (e.g., Bacterial Leaf Blight, Brown Spot, Rice Hispa) and healthy leaves.
*   **Beautiful UI/UX:** A state-of-the-art UI with glassmorphism, smooth animations, and instantaneous diagnostic feedback.
*   **Dynamic Intelligence:** Recommends immediate actionable treatments and precautions dynamically based on the detected crop ailment.

---

## 🚀 How to Run the Project Locally

Follow these instructions to set up and run PaddyGuard AI on your own computer:

### 1. Prerequisites
Make sure you have **Python 3.x** installed. You can download it from [python.org](https://www.python.org/).

### 2. Clone the Repository
Open your terminal or command prompt and run:
```bash
git clone https://github.com/nithinhv1503/paddyguard-ai.git
cd paddyguard-ai
```

### 3. Install Dependencies
Install all required Python libraries (like Flask, TensorFlow, and OpenCV) by running:
```bash
pip install -r requirements.txt
```

### 4. Provide the Trained Model & Dataset
*(Note: Large files like the `.h5` model and the image datasets are explicitly ignored by Git to prevent repository bloat).*
*   **If you want to train your own model:** 
    1. Create a `dataset/` folder in the root directory. 
    2. Place your classification folders (e.g., `Bacterial Leaf Blight`, `Rice Hispa`) filled with images inside it.
    3. Run `python train_model.py` and wait for the epochs to finish. It will automatically generate `paddy_disease_model.h5` and `class_indices.json`.
*   **If you already have the dataset but not the model:** 
    1. Download the `paddy_disease_model.h5` file from the [Releases Page](https://github.com/nithinhv1503/paddyguard-ai/releases/tag/v1.0.0) of this repository.
    2. Place it securely into the root folder of this project alongside `app.py`.

### 5. Start the Web App
Once the dependencies are installed and the model is in place, run the server:
```bash
python app.py
```

### 6. Detect Diseases!
1. Open your web browser and go to `http://127.0.0.1:5000`.
2. Navigate to the **Detect Disease** page.
3. Drag and drop (or browse) a paddy leaf image.
4. Click **Detect Disease** and view the AI's prediction and recommended treatments!
