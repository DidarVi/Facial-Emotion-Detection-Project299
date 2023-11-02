from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
from keras.models import load_model
from flask_cors import CORS

app = Flask(__name__, template_folder='templates')
CORS(app)  # Enable CORS for the entire app

# Load the pre-trained emotion recognition model
emotion_model = load_model('augmented_fer2013_emotion_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():
    try:
        # Receive image data as a file
        image_file = request.files['image']

        if image_file:
            # Read and preprocess the image
            img = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (48, 48))
            input_image = np.expand_dims(img, axis=-1)

            # Make predictions using the loaded model
            predictions = emotion_model.predict(np.expand_dims(input_image, axis=0))

            # Get the predicted emotion label
            emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
            predicted_emotion = emotion_labels[np.argmax(predictions)]

            response = {
                "predicted_emotion": predicted_emotion
            }

            return jsonify(response)  # Return a valid JSON response
        else:
            return jsonify({"error": "No image data received"}), 400  # Return a JSON error response
    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Return a JSON error response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
