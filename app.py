import os
import tempfile
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import keras_ocr
import matplotlib.pyplot as plt

app = Flask(__name__)

# Initialize the Keras OCR pipeline
pipeline = keras_ocr.pipeline.Pipeline()

# Define a temporary upload folder
UPLOAD_FOLDER = tempfile.mkdtemp()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to perform text recognition on uploaded image
def recognize_text(image_path):
    image = keras_ocr.tools.read(image_path)
    predictions = pipeline.recognize([image])
    return predictions[0] if predictions else []

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Save the uploaded file to the temporary folder
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            # Perform text recognition
            predictions = recognize_text(filename)

            # Render the results page
            return render_template('results.html', filename=filename, predictions=predictions)
    return render_template('upload.html')

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
