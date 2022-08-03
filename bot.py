import tensorflow as tf
from telegram.ext.updater import Updater
from telegram.update import Update
from telegram.ext.commandhandler import CommandHandler
from telegram.ext.messagehandler import MessageHandler
from telegram.ext.filters import Filters
from telegram.ext.callbackcontext import CallbackContext
import numpy as np
import json
from io import BytesIO


updater = Updater('5102726911:AAE9LKcgSqVRFMSThQJZvAgwLT2LLv8bdcI', use_context=True)
dogBreed_model = tf.keras.models.load_model("myModel")
file_labels = open('labels.json')
labels = json.load(file_labels)


def start(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Send me the photo of a dog and I`ll tell you its breed!;)")


def reply(update, context):
    file = context.bot.get_file(update.message.photo[-1].file_id)
    f = BytesIO(file.download_as_bytearray())
    decoded = tf.io.decode_image(f.read(), channels=3)
    scaled = tf.image.convert_image_dtype(decoded, tf.float32)
    img = tf.image.resize(scaled, size=(224, 224))
    img = tf.expand_dims(img, axis=0)
    print(img.shape)
    predicted = dogBreed_model.predict(img)
    inx = np.argmax(predicted)
    update.message.reply_text(labels[inx])


updater.dispatcher.add_handler(CommandHandler('start', start))
updater.dispatcher.add_handler(MessageHandler(Filters.photo, reply))

updater.start_polling()
