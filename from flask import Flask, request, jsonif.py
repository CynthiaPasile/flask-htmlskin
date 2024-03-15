from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import base64

app = Flask(__name__)

# Load TensorFlow model
model = tf.saved_model.load('C:/Users/cpasile/Documents/Eczema Photos TensorFlow/saved_model.pb', tags=['serve'])

# Preprocess image function
def preprocess_image(image_b64):
    # Convert base64 string to numpy array
    image = tf.image.decode_jpeg(base64.b64decode(image_b64), channels=3)
    # Resize and preprocess the image
    image = tf.image.resize(image, (224, 224)) / 255.0
    return image

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image data from request
        image_b64 = request.form['image']
        print("Received image data:", image_b64)  # Debug
        # Preprocess image
        image = preprocess_image(image_b64)
        # Make prediction
        prediction = model.predict(np.array([image]))
        print("Model predictions:", prediction)  # Debug
        # Return prediction result
        return jsonify({'prediction': prediction_result})
    except Exception as e:
        print("Prediction error:", e)  # Debug
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
