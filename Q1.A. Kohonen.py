from time import time
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import sys


class Kohonen(object):
    def __init__(self, input_vector, epochs, lr):
        random.seed(time())
        self.input_vector = input_vector
        self.map = np.random.uniform(0, 1, size=(int(math.sqrt(len(self.input_vector))), int(math.sqrt(len(self.input_vector))), 3))
        self.i_finded = -1
        self.j_fined = -1
        self.initial_lr = lr
        self.lr = lr
        self.initial_radius = 20
        self.epochs = epochs
        self.landa = self.epochs / np.log(self.initial_radius)
        self.radius = 0
        self.lr = 0
        self.compute_variables()

    def compute_variables(self, epoch=0):
        self.radius = self.initial_radius * np.exp(-epoch / self.landa)
        self.lr = self.initial_lr * np.exp(-epoch / self.landa)

    def winner(self, x):
        dist = sys.maxsize
        for i in range(int(math.sqrt(len(self.input_vector)))):
            for j in range(int(math.sqrt(len(self.input_vector)))):
                if np.sqrt(np.sum((self.map[i][j] - x) ** 2)) < dist:
                    dist = np.sqrt(np.sum((self.map[i][j] - x) ** 2))
                    self.i_finded = i
                    self.j_finded = j

    def update_weights(self, x):
        for i in range(len(self.map)):
            for j in range(len(self.map[i])):
                dist = math.sqrt((i - self.i_finded) ** 2 + (j - self.j_finded) ** 2)
                if dist < self.radius:
                    h = np.exp(-dist / (2 * (self.radius ** 2)))
                    self.map[i][j] += self.lr * h * (x - self.map[i][j])
                    
    def processing(self, x):
        self.winner(x)
        self.update_weights(x)
        


fig, axs = plt.subplots(2)
data = np.ndarray((1600, 3), dtype=float)
for i in range(len(data)):
    r = np.random.randint(0, 255)
    g = np.random.randint(0, 255)
    b = np.random.randint(0, 255)
    data[i] = [r, g, b]
data = data / data.max()

lr = 0.04
radius = 255
epochs = 3000
k = Kohonen(data, epochs, lr)
axs[0].imshow(k.map)

for e in range(epochs):
  k.compute_variables(e)
  k.processing(data[e % 1600])
  
axs[1].imshow(k.map)

