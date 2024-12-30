
import os
import numpy as np
import json
# Keras
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow_hub as hub


# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import tensorflow_hub as hub


# Define a flask app
app = Flask(__name__)
from tensorflow.keras.models import load_model

# Load the .h5 model
model = load_model('E:\\Brain-Tumor-Classification-System-main\\braintumor.h5', custom_objects={'KerasLayer': hub.KerasLayer})

# Save it as SavedModel
model.save('E:\\Brain-Tumor-Classification-System-main\\saved_model', save_format='tf')


# Model saved with Keras model.save()
try:
    model = tf.keras.models.load_model('E:\\Brain-Tumor-Classification-System-main\\saved_model')


except Exception as e:
    print(f"Error loading model: {e}")


def model_predict(img_path, model):
    test_image = image.load_img(img_path, target_size = (150,150))
    test_image = image.img_to_array(test_image)
    test_image = np.array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)

    labels = ['Glioma','Meningioma','No Tumor','Pituitary']

    return labels[result.argmax()] , result


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds , res = model_predict(file_path, model)

        data = {
            "pred" : preds,
            "glioma" : str(round(res[0][0]*100,4)),
            "meningioma" :str(round(res[0][1]*100,4)),
            "notumor": str(round(res[0][2]*100,4)),
            "pituitary":str(round(res[0][3]*100,4)),
            "glink": "https://www.lybrate.com/topic/glioma",
            "mlink" : "https://www.lybrate.com/topic/meningioma",
            "nlink" : "https://www.aans.org/en/Patients/Neurosurgical-Conditions-and-Treatments/Brain-Tumors",
            "plink" : "https://medsurgeindia.com/cost/pituitary-tumor-treatment-cost-in-india/#:~:text=A%20pituitary%20tumor%20is%20a,that%20regulate%20vital%20bodily%20processes."
        }

        y = json.dumps(data)
        return y
    return None


if __name__ == '__main__':
    app.run(debug=False)
