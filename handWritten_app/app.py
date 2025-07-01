from flask import Flask, render_template, request
import numpy as np
import cv2
import pickle

app = Flask(__name__)

# Load PCA and KNN model
pca = pickle.load(open('model/pca.pkl', 'rb'))
model = pickle.load(open('model/model.pkl', 'rb'))

# Preprocess and apply PCA
def preprocess_image(image_file):
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))                 # Same size as training images
    image = 16 - (image // 16)                        # Normalize to 0–16
    image = image.reshape(1, -1)                      # Shape: (1, 64)
    image_pca = pca.transform(image)                  # Apply same PCA
    return image_pca

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded"

    file = request.files['image']
    if file.filename == '':
        return "No file selected"

    img = preprocess_image(file)
    prediction = model.predict(img)[0]
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)