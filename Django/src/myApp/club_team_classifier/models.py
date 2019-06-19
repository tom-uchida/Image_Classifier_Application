from django.db import models

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from PIL import Image
import io, base64

graph = tf.get_default_graph()

class Photo(models.Model):
    image = models.ImageField(upload_to='photos')

    IMAGE_SIZE = 224
    MODEL_FILE_PATH = "./club_team_classifier/ml_models/vgg16_transfer.h5"

    classes = [ "Barcelona", 
                "BayernMunchen", 
                "Chelsea", 
                "Dortmund", 
                "Juventus", 
                "Liverpool", 
                "ManchesterCity", 
                "ManchesterUnited", 
                "Milan",
                "RealMadrid"]
    num_of_classes = len(classes)

    def predict(self):
        model = None
        global graph
        with graph.as_default():
            # Load model file(.h5 file)
            model = load_model(self.MODEL_FILE_PATH)

            img_data = self.image.read()
            img_bin  = io.BytesIO(img_data)

            image = Image.open(img_bin)
            image = image.convert("RGB")
            image = image.resize( (self.IMAGE_SIZE, self.IMAGE_SIZE) )
            image_np = np.asarray(image) / 255.0
            X = []
            X.append(image_np)
            X = np.array(X)

            result = model.predict([X])[0]
            predicted_index = result.argmax()
            percentage = int(result[predicted_index] * 100)

            # print("\n", self.classes[predicted_index], ":", percentage, "%\n")

            return self.classes[predicted_index], percentage