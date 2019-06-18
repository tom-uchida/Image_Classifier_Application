import numpy as np
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.applications import VGG16

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

# Load .npy file
X_train, X_test, t_train, t_test = np.load("./train_and_test_data_224.npy", allow_pickle=True)
# Convert label to one-hot vector
t_train = np_utils.to_categorical(t_train, num_of_classes)
t_test  = np_utils.to_categorical(t_test, num_of_classes)
# Normalize
X_train = X_train.astype("float") / 255.0
X_test  = X_test.astype("float") / 255.0

# Define model
model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
# print("Model loaded.")
# model.summary()

# Fully-connected
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(num_of_classes, activation="softmax"))

model = Model(inputs=model.input, outputs=top_model(model.output))
model.summary()

for layer in model.layers[:15]:
    layer.trainable = False


opt = Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
model.fit(X_train, t_train, batch_size=32, epochs=10)
score = model.evaluate(X_test, t_test, batch_size=32)

# Save model
model.save("./vgg16_transfer.h5")
print("Model saved.")