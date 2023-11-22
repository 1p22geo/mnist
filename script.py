import tensorflow as tf
import cv2 as cv
import numpy as np

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(96, activation='relu'),
  tf.keras.layers.Dense(96, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])


model.fit(x_train, y_train, epochs=100)

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

while True:
    try:
        filename = input("what file to read: ")
        img = cv.imread(filename)
        gr = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        res = cv.resize(gr, (28,28), interpolation=cv.INTER_CUBIC)
        
        
        def f(x):
            return (255-x)/256
        
        image = np.array([[f(x) for x in arr] for arr in res])
        
        res = probability_model.predict(np.array([image]))[0]

        out = {}
        for number in range(10):
            out[str(number)] = str( int(res[number] * 10000)/100 ) + "%"

        for k,v in out.items():
            print(f"{k}-> {v}")
    except Exception as e:
        print(e)
