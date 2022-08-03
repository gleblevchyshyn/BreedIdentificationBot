from CreatingDataSet import one_hot, get_processed_imgs, Y
import tensorflow as tf
import tensorflow_hub as hub
import json
from sklearn.model_selection import train_test_split

# preprocessing data that model can understand it
f = open('labels.json')
labels_ = json.load(f)
y = Y['breed'].apply(lambda x: one_hot(x, labels_)).tolist()
X = get_processed_imgs('train.zip', 224)
y = tf.convert_to_tensor(y)
X = tf.convert_to_tensor(X, dtype=tf.float32)

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

MODEL_URL = "https://tfhub.dev/tensorflow/resnet_50/classification/1"

# Creating the model for ResNet50V2
model = tf.keras.Sequential([
    # Layer 1 : Input Layer
    hub.KerasLayer(MODEL_URL),

    # Layer 2 : Output Layer
    tf.keras.layers.Dense(120, activation='softmax')
])

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
model.build((None, 224, 224, 3))
model.summary()

# Fitting the model
model.fit(x=X, y=y, epochs=10, batch_size=32)
model.save("myModel")
