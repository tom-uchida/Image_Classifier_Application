from PIL import Image
import os, glob
import numpy as np
from sklearn import model_selection

# Initialize parameter
# classes = [ "Arsenal", 
#             "AtleticoMadrid", 
#             "Barcelona", 
#             "BayernMunchen", 
#             "Chelsea", 
#             "Dortmund", 
#             "Inter", 
#             "Juventus", 
#             "Liverpool", 
#             "ManchesterCity", 
#             "ManchesterUnited", 
#             "Milan", 
#             "PSG", 
#             "RealMadrid", 
#             "Tottenham"]
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
image_size = 150
num_of_testdata = 10

# Read image and convert to NumPy array
X_train = [] # image
X_test  = []
t_train = [] # index
t_test  = []

for class_index, class_name in enumerate(classes):
    # Read class images
    images_dir = "../images/" + class_name
    images = glob.glob(images_dir + "/*.jpg")

    # Read images one by one
    for image_index, target_image in enumerate(images):
        # Read only 50 images
        if image_index >= 50: break
        image = Image.open(target_image)
        image = image.convert("RGB")
        image = image.resize( (image_size, image_size) )

        # Store 10 test images
        if image_index <= num_of_testdata:
            image_np = np.asarray(image)
            X_test.append( image_np )
            t_test.append(class_index)

        # Increase the image
        else:
            for angle in range(-20, 20, 20):
                # Rotation
                image_rot = image.rotate(angle)
                image_rot_np = np.asarray(image_rot)
                X_train.append( image_rot_np )
                t_train.append(class_index)

                # Reverse
                image_trans = image_rot.transpose(Image.FLIP_LEFT_RIGHT)
                image_trans_np = np.asarray(image_trans)
                X_train.append( image_trans_np )
                t_train.append(class_index)
            # end for angle
    # end for image_index
# end for class_index

# Convert list to numpy array
X_train, X_test = np.array(X_train), np.array(X_test)
X_train, X_test = X_train / 255.0, X_test / 255.0
t_train, t_test = np.array(t_train), np.array(t_test)

# Separete data into training data and test data
xt = (X_train, X_test, t_train, t_test)
np.save("train_and_test_data.npy", xt)