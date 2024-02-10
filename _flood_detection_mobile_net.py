# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout
from keras.applications import VGG19
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from google.colab import drive

# Mount Google Drive
drive.mount('drive')

import os
import shutil
import random
import itertools
# %matplotlib inline
import numpy as np
import tensorflow as tf
import matplotlib as mpl
from keras import backend
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from keras.applications import imagenet_utils
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Activation
from sklearn.metrics import precision_score, recall_score
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import decode_predictions, preprocess_input
import requests
import urllib.request
from PIL import Image

from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.models import Model

labels = ['Flooding', 'No Flooding']
train_path = 'train_df'
valid_path = 'valid_df'
test_path = 'test_df'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=train_path, target_size=(224,224), batch_size=10) #preprocess_input is used for mobilenet
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=valid_path, target_size=(224,224), batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=test_path, target_size=(224,224), batch_size=10, shuffle=False)

mobile = tf.keras.applications.mobilenet.MobileNet(weights='imagenet', include_top=False)
#initial weights taken from the imagenet dataset
#final layers are not included for the cnn

x = mobile.layers[-12].output
x

# Create global pooling, dropout and a binary output layer, as we want our model to be a binary classifier,
# i.e. to classify flooding and no flooding
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
output = Dense(units=2, activation='sigmoid')(x)

# Construct the new fine-tuned mode
model = Model(inputs=mobile.input, outputs=output)

# Freez weights of all the layers except for the last five layers in our new model,
# meaning that only the last 12 layers of the model will be trained.
for layer in model.layers[:-23]:
    layer.trainable = False

model.summary()

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=train_batches,
          steps_per_epoch=len(train_batches),
          validation_data=valid_batches,
          validation_steps=len(valid_batches),
          epochs=10,
          verbose=2
)

# Saving and loading our trained for future use

model.save("fine_tuned_flood_detection_model")
# model.load_weights('fine_tuned_flood_detection_model')

# Make predictions and plot confusion matrix to look how well our model performed in classifying
# flooding and no flooding images

test_labels = test_batches.classes
predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)
cm = confusion_matrix(y_true=test_labels, y_pred=predictions.argmax(axis=1))
precision = precision_score(y_true=test_labels, y_pred=predictions.argmax(axis=1))
f1_score = f1_score(y_true=test_labels, y_pred=predictions.argmax(axis=1))
accuracy = accuracy_score(y_true=test_labels, y_pred=predictions.argmax(axis=1))
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# # Pring precision, F1 score and accuracy of our model
# print('Precision%: ', precision*100)
# print('F1 Score%: ', f1_score*100)
# print('Accuracy%: ', accuracy*100)

# Print precision, F1 score, and accuracy of our model in percentages with 3 decimal places
print('Precision: {:.2f}'.format(precision * 100))
print('F1 Score: {:.2f}'.format(f1_score * 100))
print('Accuracy: {:.2f}'.format(accuracy * 100))

# Confusion Matrix
test_batches.class_indices
cm_plot_labels = ['Flooding','No Flooding']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

"""Evaluate our finetuned model"""

# Display image which we want to predict
from IPython.display import Image
Image(filename='zfelvjc052cu2cxy37i6dgi11045iynv.jpeg', width=300,height=200)

from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
model = tf.keras.models.load_model("fine_tuned_flood_detection_model")

# Define the path to the image you want to test
image_path = 'zfelvjc052cu2cxy37i6dgi11045iynv.jpeg'

# Load and preprocess the image
img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = tf.keras.applications.mobilenet.preprocess_input(img_array)

# Make predictions
predictions = model.predict(img_array)

# Interpret the predictions
labels = ['Flooding', 'No Flooding']
predicted_label = labels[np.argmax(predictions)]

print("Predicted label:", predicted_label)
