import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils

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
image_size = 150

# Load .npy file
X_train, X_test, t_train, t_test = np.load("./train_and_test_data.npy")
# Convert label to one-hot vector
t_train = np_utils.to_categorical(t_train, num_of_classes)
t_test = np_utils.to_categorical(t_test, num_of_classes)

# Define model
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(image_size, image_size, 3)))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_of_classes, activation='softmax'))

# opt = SGD(lr=0.01) # rmsprop, adam
opt = Adam()
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

model.fit(X_train, t_train, batch_size=32, epochs=10)

score = model.evaluate(X_test, t_test, batch_size=32)