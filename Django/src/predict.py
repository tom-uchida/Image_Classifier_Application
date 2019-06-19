import numpy as np
from tensorflow import keras
from keras.models import Sequential, Model, load_model
from PIL import Image
import sys

# Initialize parameter
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
image_size = 224

# Read image from command line argument
image = Image.open(sys.argv[1])
image = image.convert("RGB")
image = image.resize( (image_size, image_size) )
image_np = np.asarray(image) / 255.0
X = []
X.append(image_np)
X = np.array(X)

# Load model(.h5 file)
model = load_model("./vgg16_transfer.h5")

# Get result
result = model.predict([X])[0]
predicted_index = result.argmax()
percentage = int(result[predicted_index] * 100)

print("\n", classes[predicted_index], ":",percentage, "%\n")