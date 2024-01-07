import numpy as np
import os
from Layer_Dense import Layer_Dense
from Activation_Relu import Activation_ReLU
from LossFunction import Loss_CategoricalCrossentropy
from Softmax import Activation_Softmax
from Optimizer import Optimizer_SGD
from tensorflow import keras as data
from PIL import Image
from DigitRecGUI import DigitRecGUI
from Model import Model
import tkinter as tk


layers = [(784, 128), (128, 64), (64, 10)]  # Example layers
batch_size = 32
epochs = 10
model = Model(layers, batch_size, epochs)
model.load_data(data.datasets.mnist.load_data())
model.train()
model.evaluate(model.X_test, model.Y_test)  # Evaluate on test set
model.save_model(version=1)
model.load_model(version=1)