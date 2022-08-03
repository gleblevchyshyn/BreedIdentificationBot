import tensorflow as tf

model = tf.keras.models.load_model("myModel")

model.summary()