from PIL import Image
import os, glob
import numpy as np
from sklearn import model_selection

classes = ["building", "person", "tree"]
num_of_classes = len(classes)
image_size = 50
num_of_testdata = 100

# Read images
X_train = [] # class images
X_test  = []
Y_train = [] # class indexes
Y_test  = []
for class_index, class_label in enumerate(classes):
    images_dir = "../images/" + class_label
    images = glob.glob(images_dir + "/*.jpg")
    for i, target_image in enumerate(images):
        if i >= 200: break
        image = Image.open(target_image)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        image_data = np.asarray(image)

        if i < num_of_testdata:
            X_test.append(image_data)
            Y_test.append(class_index)
        else:
            X_train.append(image_data)
            Y_train.append(class_index)

            # Increase the image
            for angle in range(-20, 20, 5):
                # Rotation
                image_rot = image.rotate(angle)
                image_data = np.asarray(image_rot)
                X_train.append(image_data)
                Y_train.append(class_index)

                # Reverse
                image_trans = image.transpose(Image.FLIP_LEFT_RIGHT)
                image_data = np.asarray(image_trans)
                X_train.append(image_data)
                Y_train.append(class_index)
            # end for angle
    # end for image
# end for class

# Convert List to NumPy array
# X = np.array(X)
# Y = np.array(Y)
X_train, X_test = np.array(X_train), np.array(X_test)
y_train, y_test = np.array(Y_train), np.array(Y_test)

# Separate data into training data and test data
# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y) # (3:1)
xy = (X_train, X_train, y_train, y_test)
np.save("./image_data_aug.npy", xy)