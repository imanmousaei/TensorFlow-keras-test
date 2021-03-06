import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.python.keras.models import load_model

trainingSamples = []
trainingLabels = []

# [0,499] -> 0 , [500,999] -> 1

for i in range(0, 500):
    trainingSamples.append(i)
    trainingLabels.append(0)

for i in range(500, 1000):
    trainingSamples.append(i)
    trainingLabels.append(1)

trainingSamples = np.array(trainingSamples)
trainingLabels = np.array(trainingLabels)

# shuffles numpy arrays respective to each other :
# trainingSamples, trainingLabels = shuffle(trainingSamples, trainingLabels)

# [0,999] -> [0,1] & transforming it to 2D form ( for fit function )
scaler = MinMaxScaler(feature_range=(0, 1))
scaledTrainingSamples = scaler.fit_transform(trainingSamples.reshape(-1, 1))

for num in scaledTrainingSamples:
    print(num)

# creates neural network ( starting from 2nd layer cuz 1st layer is out numpy array trainingSamples )
# Dense = fully connected layer
model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),  # todo input_shape??
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])

# model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# batch_size : how many samples are processed by NN simultaneously
# epochs : how many times all samples are processed by NN
# verbose : log level
# validation_split : splits the last 20% of the trainingSamples for validation ( & 80% of it for training )
model.fit(x=scaledTrainingSamples, y=trainingLabels, validation_split=0.2, batch_size=10, epochs=30, shuffle=True, verbose=2)  # trains NN

# testing NN :
testSet = np.array([1, 100, 780, 900, 1010, -100, 500])
testAnswer = np.array([0, 0, 1, 1, 1, 0, 1])
testSet = scaler.fit_transform(testSet.reshape(-1, 1))


print()
test = model.evaluate(x=testSet, y=testAnswer)

# save the network :
model.save('../neural_network_models/test_model.h5')

# load the network :
new_model = load_model('../neural_network_models/test_model.h5')
