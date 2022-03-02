import os
import pandas as pd

from model import TransferModel

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from keras.optimizers import adam_v2

# Global settings
INPUT_DATA_DIR = 'data/cars_filtered_top300/'
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
TARGET = 'model'
BASE = 'ResNet'

# All available training images
files = [file for file in os.listdir(INPUT_DATA_DIR) if file.endswith(".jpg")]
file_paths = [INPUT_DATA_DIR + file for file in files]

# Create a list of all possible outcomes
if TARGET == 'make':
    classes = list(set([file.split('_')[0] for file in files]))
if TARGET == 'model':
    classes = list(set([file.split('_')[0] + '_' + file.split('_')[1] for file in files]))

# Targets in list
classes_lower = [x.lower() for x in classes]

# Split paths into train, valid, and test
files_train, files_test = train_test_split(file_paths, test_size=0.25)
files_train, files_valid = train_test_split(files_train, test_size=0.25)


# Show examples from one batch
plot_size = (18, 18)

# Init base model and compile
model = TransferModel(base=BASE,
                      shape=INPUT_SHAPE,
                      classes=classes,
                      unfreeze='all')

model.compile(loss="categorical_crossentropy",
              optimizer=adam_v2.Adam(0.0001),
              metrics=["categorical_accuracy"])

class_weights = compute_class_weight('balanced', classes, pd.Series([file.split('_')[0] + "_" + file.split('_')[1] for file in files]))

# Train model using defined tf.data.Datasets
model.history = model.train(ds_train=ds_train, ds_valid=ds_valid, epochs=10, class_weights=class_weights)

# Plot accuracy on training and validation data sets
model.plot()

# Evaluate performance on testing data
model.evaluate(ds_test=ds_test)