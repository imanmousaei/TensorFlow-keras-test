import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

trainingSamples = []
trainingLabels = []

# [0,499] -> 0 , [500,999] -> 1

for i in range(1, 500):
    trainingSamples.append(i)
    trainingLabels.append(0)

for i in range(500, 1000):
    trainingSamples.append(i)
    trainingLabels.append(1)

trainingSamples = np.array(trainingSamples)
trainingLabels = np.array(trainingLabels)

# shuffles numpy arrays respective to each other :
trainingSamples, trainingLabels = shuffle(trainingSamples, trainingLabels)

# [1,1000] -> [0,1] & transforming it to 2D form ( for fit function )
scaler = MinMaxScaler(feature_range=(0, 1))
scaledTrainingSamples = scaler.fit_transform(trainingSamples.reshape(-1, 1))

"""
for num in scaledTrainingSamples:
    print(num)
"""

