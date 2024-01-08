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


layers = [(784, 256), (256, 64), (64, 10)]  # Example layers
batch_size = 32
epochs = 18
model = Model(layers, batch_size, epochs)
# model.load_data(data.datasets.mnist.load_data())
# model.train()
# model.evaluate(model.X_test, model.Y_test)  # Evaluate on test set
# model.save_model(version=16)
model.load_model(version=16)
def predict_callback(image, label):
        prediction = model.predict(gui.image)
        gui.display_prediction(prediction)

    # Define the train model callback
def train_model_callback():
     model.load_TrainData('data.npz')
     model.train()
    # model.evaluate(model.X_test, model.Y_test)  # Evaluate on test set
     model.save_model(version=16)

    # Initialize your GUI
window = tk.Tk()
gui = DigitRecGUI(window, predict_callback, train_model_callback)
window.mainloop()