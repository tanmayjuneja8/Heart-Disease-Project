import pandas as pd
import numpy as np
import tensorflow as tf

w = tf.Variable(0, dtype=tf.float32)
x = np.array([1.0, -10.0, 25.0], dtype=np.float32)
optimizer = tf.keras.optimizers.Adam(0.1)


def training_classifier(x, w, optimizer):
    def train_step():
        return x[0] * w ** 2 + x[1] * w + x[2]

    for i in range(1000):
        optimizer.minimize(train_step, [w])
    return w


w = training_classifier(x, w, optimizer)
print(w)
