from PIL import Image
import os, glob
import numpy as np
from sklearn import model_selection

# Initialize parameter
classes = [ "Arsenal", 
            "AtleticoMadrid", 
            "Barcelona", 
            "BayernMunchen", 
            "Chelsea", 
            "Dortmund", 
            "Inter", 
            "Juventus", 
            "Liverpool", 
            "ManchesterCity", 
            "ManchesterUnited", 
            "Milan", 
            "PSG", 
            "RealMadrid", 
            "Tottenham"]
num_classes = len(classes)
image_size = 150

# Read image and convert to NumPy array
X = [] # list
T = [] # list

for index, classlabel in enumerate(classes):
    images_dir = "../images/" + classlabel
    images = glob.glob(images_dir + ".jpg")
    for i, target_image in enumerate(images):
        image = Image.open(target_image)
        image = image.convert("RGB")
        image = image.resize(image_size, image_size)
        X.append( np.asarray(image) )
        T.append(index)
    # end for i
# end for index

# Convert list to numpy array
X = np.array(X)
T = np.array(T)

X_train, X_test, t_train, t_test = model_selection.train_test_split(X, T)
xt = (X_train, X_test, t_train, t_test)
np.save("./image_data,npy", xt)