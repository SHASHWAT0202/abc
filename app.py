from flask import Flask, request, jsonify, render_template, url_for
import cv2
import numpy as np
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Initialize Flask app
app = Flask(__name__)

# Load the trained emotion detection model
model = load_model("emotion_model.h5")

# Define emotion labels (ensure this matches the dataset labels)
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Function to preprocess the input image
def preprocess_image(image):
    # Convert to grayscale for the model
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return None, "No face detected. Ensure your face is clearly visible."

    # Focus on the first detected face
    x, y, w, h = faces[0]
    face = gray_image[y:y+h, x:x+w]

    # Resize to 48x48 pixels
    resized_face = cv2.resize(face, (48, 48))

    # Normalize and reshape for the model
    normalized_face = resized_face / 255.0
    reshaped_face = np.expand_dims(img_to_array(normalized_face), axis=0)

    return reshaped_face, None

# Function to predict emotion
def predict_emotion(image):
    preprocessed_image, error = preprocess_image(image)
    if error:
        return None, error

    predictions = model.predict(preprocessed_image)
    emotion_index = np.argmax(predictions)
    return emotion_labels[emotion_index], None

# Routes for rendering HTML pages
@app.route('/')
def home():
    return render_template('index.html')  # Frontend HTML for the home page


@app.route("/analyze", methods=["POST"])
def analyze_mood():
    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    mood, error = predict_emotion(image)

    if error:
        response = {
            "mood": "Error",
            "recommendations": [error]
        }
    else:
        recommendations = {
            "Happy": ["Keep smiling!", "Celebrate your happiness!", "Spread the joy around!", "Enjoy the moment!"],
            "Sad": ["Take a walk outside.", "Talk to a trusted friend.", "Listen to your favorite music.", "Write your feelings down."],
            "Angry": ["Take a deep breath and count to 10.", "Step outside for fresh air.", "Practice mindfulness or meditation.", "Channel your energy into a creative activity."],
            "Neutral": ["Maintain your balance and focus.", "Enjoy your steady mood.", "Reflect on your day and plan ahead.", "Take time for a little self-care."],
            "Surprise": ["Embrace the unexpected moment!", "Share the surprise with others.", "Keep an open mind for what's next!", "Take a moment to process your emotions."],
            "Fear": ["Ground yourself by breathing deeply.", "Talk to someone you trust.", "Focus on what you can control.", "Remind yourself that you're safe."],
            "Disgust": ["Refocus on things you enjoy.", "Take a moment to clear your mind.", "Engage in an activity you love.", "Take a break and refresh yourself."]
        }

        response = {
            "mood": mood,
            "recommendations": random.sample(recommendations.get(mood, ["Stay positive and keep going!"]), k=2)
        }

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
