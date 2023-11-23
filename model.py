import tensorflow as tf
import cv2 as cv
import numpy as np
import threading


EPOCHS=10

class Model():
    def __init__(self):
        mnist = tf.keras.datasets.mnist

        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='sigmoid'),
            tf.keras.layers.Dense(96, activation='sigmoid'),
            tf.keras.layers.Dense(96, activation='sigmoid'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10)
        ])

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.model.compile(optimizer='adam',
                      loss=loss_fn,
                      metrics=['accuracy'])


        self.th = threading.Thread(target=self.train)
        self.th.start()

    def train(self):
        print("started training model, 60000 dataset items")
        x = 0
        while True:
            self.model.fit(self.x_train, self.y_train, epochs=EPOCHS)
            x += 10
            print(f"{x} epochs done, continuing training")
    
            
    def predict(self, filename):
        img = cv.imread(filename)
        gr = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        res = cv.resize(gr, (28,28), interpolation=cv.INTER_CUBIC)
        export_model = tf.keras.Sequential([
            self.model,
            tf.keras.layers.Softmax()
        ])

        
        def f(x):
            q = (((255-x)/256)**3)*2
            return sorted([0,q,1])[1]
        
        image = np.array([[f(x) for x in arr] for arr in res])
        
        res = export_model.predict(np.array([image]))[0]
        #out = np.argmax(res)
        out = {}
        for number in range(10):
             out[int(number)] = ( int(res[number] * 10000)/100 )
        out = dict(sorted(out.items(), key=lambda item: item[1], reverse=True))
        print(out)
        return out
