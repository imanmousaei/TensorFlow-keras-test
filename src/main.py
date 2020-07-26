import tensorflow as tf
from tensorflow import keras

trainingSamples = []
trainingLabels = []

# [1,500] -> 0 , [501,1000] -> 1

for i in range(1, 500):
    trainingSamples.append(i)
    trainingLabels.append(0)

for i in range(501, 1000):
    trainingSamples.append(i)
    trainingLabels.append(1)

