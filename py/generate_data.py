from PIL import Image
import os, glob
import numpy as np
from sklearn import model_selection

classes = ["building", "person", "tree"]
num_of_classes = len(classes)
image_size = 50

# Read images
X = []
Y = []
for class_index, class_label in enumerate(classes):
    images_dir = "../images/" + class_label
    images = glob.glob(images_dir + "/*.jpg")
    for i, target_image in enumerate(images):
        if i >= 250: break
        image = Image.open(target_image)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        image_data = np.asarray(image)
        X.append(image_data)
        Y.append(class_index)
    # end for
# end for

# Convert List to NumPy array
X = np.array(X)
Y = np.array(Y)

# Separate data into training data and test data
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y) # (3:1)
xy = (X_train, X_train, Y_train, Y_test)
np.save("./image_data.npy", xy)