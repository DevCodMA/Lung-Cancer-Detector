from flask import Flask, request, render_template
import os
import base64
from keras.models import load_model
import numpy as np
import cv2


app = Flask(__name__)

# Set the directory where uploaded images will be saved
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def predict(filename):
    try:
        img = cv2.cvtColor(cv2.imread(f"{filename}"), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        model = load_model('model/model.h5')
        prediction = model.predict(img)
        return prediction
    except Exception as ex:
        return f"{ex}"

# predict("Malignant case (38).jpg")
# Route for the home page
@app.route('/')
def home():
    return render_template('index.html', upimg='0')

# Route for image upload
@app.route('/process', methods=['POST'])
def process():
    Class_names = ['Adenocarcinoma', 'Large Cell Carcinoma', 'Normal', 'Squamous Cell Carcinoma']
    # Check if the request contains a file
    if 'image_file' not in request.files:
        return 'No file found', 400

    if request.form['isTrue'] in ['0', 0]:
        return render_template('index.html', disp=0, upimg="0", img="null")

    # Get the uploaded file
    img = request.files['image_file']
    file = img.filename
    img.save(os.path.join(app.config['UPLOAD_FOLDER'], img.filename))
    try:
        prediction = predict(f"Uploads/{file}")
        with open(os.path.join(app.config['UPLOAD_FOLDER'], file), 'rb') as f:
            img2 = f.read()
        dataURL = f"data:jpeg;base64,{base64.b64encode(img2).decode('utf-8')}"
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))
        data = np.argmax(prediction)
        print(Class_names[data])
        return render_template('index.html', disp=1, detect=data, upimg=dataURL, res=Class_names[data])
    except Exception as ex:
        print(f"Exception Raised: {ex}")
        return render_template('index.html', disp=0, upimg="0")

if __name__ == '__main__':
    app.run(debug=True)