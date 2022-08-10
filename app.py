import streamlit
from zipfile import ZipFile
import os,glob
import numpy as np
import tensorflow as tf
import keras
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Convolution2D, Dropout, Dense,MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from zipfile import ZipFile
import pickle
import PIL
from PIL import Image
from math import exp, tanh
from tokenize import Exponent
from enum import Enum
from io import BytesIO, StringIO
import numpy as np
import PIL
import tensorflow as tf
from tensorflow import keras
from datetime import date
from io import BytesIO
from IPython import display
import base64
import pandas as pd
import uuid
DATADIR = r'C:\\Users\\USER\\Documents\\FNA'
CATEGORIES = ['benign', 'malignant']
for category in CATEGORIES:
 path = os.path.join(DATADIR, category)
import pathlib
data_dir = pathlib.Path(DATADIR)
image_count = len(list(data_dir.glob('*/*.png')))
benign = list(data_dir.glob('benign/*'))
malignant = list(data_dir.glob('malignant/*'))
batch_size = 32
img_height = 180
img_width = 180
train_ds = tf.keras.utils.image_dataset_from_directory(
 data_dir,
 validation_split=0.2,
 subset="training",
 seed=123,
 image_size=(img_height, img_width),
 batch_size= 32)
val_ds = tf.keras.utils.image_dataset_from_directory(
 data_dir,
 validation_split=0.2,
 subset="validation",
 seed=123,
 image_size=(img_height, img_width),
 batch_size=32)
class_names = train_ds.class_names
print(class_names)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
normalization_layer = layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)
)
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
num_classes = len(class_names)
model = Sequential([
 layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
 layers.Conv2D(16, 3, padding='same', activation='relu'),
 layers.MaxPooling2D(),
 layers.Conv2D(32, 3, padding='same', activation='relu'),
 layers.MaxPooling2D(),
 layers.Conv2D(64, 3, padding='same', activation='relu'),
 layers.MaxPooling2D(),
 layers.Flatten(),
 layers.Dense(128, activation='relu'),
 layers.Dense(num_classes)
])
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.summary()
epochs=10
history = model.fit(
 train_ds,
 validation_data=val_ds,
 epochs=epochs
)
data_augmentation = keras.Sequential(
 [
 layers.RandomFlip("horizontal",
 input_shape=(img_height,
 img_width,
3)),
 layers.RandomRotation(0.1),
 layers.RandomZoom(0.1),
 ]
)
model = Sequential([
 data_augmentation,
 layers.Rescaling(1./255),
 layers.Conv2D(16, 3, padding='same', activation='relu'),
 layers.MaxPooling2D(),
 layers.Conv2D(32, 3, padding='same', activation='relu'),
 layers.MaxPooling2D(),
 layers.Conv2D(64, 3, padding='same', activation='relu'),
 layers.MaxPooling2D(),
 layers.Dropout(0.2),
 layers.Flatten(),
 layers.Dense(128, activation='relu'),
 layers.Dense(num_classes)
])
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.summary()
epochs = 15
history = model.fit(
 train_ds,
 validation_data=val_ds,
 epochs=epochs
)
file = streamlit.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if file is not None:
 image = Image.open(file)
 streamlit.image(image)
 img_array = np.array(image)
 img = tf.image.resize(img_array, size=(180,180))
 img = tf.expand_dims(img, axis=0)
 predictions = model.predict(img)
 score = tf.nn.softmax(predictions[0])
 streamlit.title(
 "This image is most likely {} with a {:.2f} percent confidence."
 .format(class_names[np.argmax(score)], 100 * np.max(score))
 )