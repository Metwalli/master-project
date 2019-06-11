# organize imports
from __future__ import print_function

# import pandas as pd
from keras.models import Model
from keras.optimizers import Adam
# from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.densenet import DenseNet121
from keras.layers import Flatten, Input, Dense
# from sklearn.metrics import classification_report
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix
# import numpy as np
# import h5py
import argparse
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
import time
import os

# from dense_inception import DenseNetInception
from dense_inception_concat import DenseNetInceptionConcat, DenseNetBaseModel, DenseNetInception
from utils import Params
# import pickle
# import seaborn as sns
# import matplotlib.pyplot as plt


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data_dir", required=False,
                help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model_dir", required=False,
                help="path to Model config (i.e., directory of Model)")
ap.add_argument("-r", "--restore_from", required=False,
                help="path of saved checkpoints (i.e., directory of check points)")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions


# Arguments
data_dir = args["data_dir"]
model_dir = args["model_dir"]
restore_from = args["restore_from"]


# load the user configs

params = Params(os.path.join(model_dir, 'params.json'))

# config variables
EPOCHS = params.num_epochs
INIT_LR = params.learning_rate
BS = params.batch_size
IMAGE_DIMS = (params.image_size, params.image_size, 3)
seed      = 2019
results     = params.results_path
classifier_path = params.classifier_path
model_path = params.model_path
model_name = params.model_name
use_imagenet_weights = params.use_imagenet_weights


if data_dir is None:
    data_dir = params.data_dir

# Dataset Directory
train_dir = os.path.join(data_dir, "train")
valid_dir = os.path.join(data_dir, "test")
train_datagen = ImageDataGenerator(rotation_range=25,
                                   width_shift_range=0.1,
                                   rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMAGE_DIMS[1], IMAGE_DIMS[1]),
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        valid_dir,
        target_size=(IMAGE_DIMS[1], IMAGE_DIMS[1]),
        batch_size=BS,
        class_mode='categorical')
# initialize the model
CLASSES = train_generator.num_classes

print ("[INFO] creating model...")


if model_name == 'base':
    model = DenseNetBaseModel(CLASSES, use_imagenet_weights)
elif model_name == 'inject':
    model = DenseNetInceptionConcat(num_labels=CLASSES, use_imagenet_weights=use_imagenet_weights).model
else:
    model = DenseNetInception(input_shape= IMAGE_DIMS, params=params)
model.summary()

if restore_from is not None:
    model.load_weights(os.path.join(model_dir, restore_from))

print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

print ("[INFO] training started...")

tensorBoard = TensorBoard(log_dir='logs/{}'.format(time.time()))
# checkpoint

checkpoint = ModelCheckpoint("output/ckpts_dir", monitor='val_acc', period=5, verbose=1, save_best_only=True, mode='max')
callbacks_list = checkpoint

M = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.n // train_generator.batch_size,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.n // validation_generator.batch_size,
        callbacks=[callbacks_list, tensorBoard])


# save the model to disk
print("[INFO] serializing network...")
model.save(os.path.join(model_dir, "last.weights.h5"))

"""
model.evaluate_generator(generator=validation_generator)
validation_generator.reset()
pred=model.predict_generator(validation_generator, verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames=validation_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv(os.path.join(results, "results.csv"),index=False)
"""
"""
# use rank-1 and rank-5 predictions
print ("[INFO] evaluating model...")
f = open(results, "w")
rank_1 = 0
rank_5 = 0

# loop over test data
for (label, features) in zip(testLabels, testData):
    # predict the probability of each class label and
    # take the top-5 class labels
    predictions = model.predict_proba(np.atleast_2d(features))[0]
    predictions = np.argsort(predictions)[::-1][:5]

    # rank-1 prediction increment
    if label == predictions[0]:
        rank_1 += 1

    # rank-5 prediction increment
    if label in predictions:
        rank_5 += 1

# convert accuracies to percentages
rank_1 = (rank_1 / float(len(testLabels))) * 100
rank_5 = (rank_5 / float(len(testLabels))) * 100

# write the accuracies to file
f.write("Rank-1: {:.2f}%\n".format(rank_1))
f.write("Rank-5: {:.2f}%\n\n".format(rank_5))

# evaluate the model of test data
preds = model.predict(testData)

# write the classification report to file
f.write("{}\n".format(classification_report(testLabels, preds)))
f.close()

# dump classifier to file
print ("[INFO] saving model...")
pickle.dump(model, open(classifier_path, 'wb'))

# display the confusion matrix
print ("[INFO] confusion matrix")

# get the list of training lables
labels = sorted(list(os.listdir(train_dir)))

# plot the confusion matrix
cm = confusion_matrix(testLabels, preds)
sns.heatmap(cm,
            annot=True,
            cmap="Set2")
plt.show()


----------------



# USAGE
# python train.py --dataset dataset --model pokedex.model --labelbin lb.pickle

# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import TensorBoard
import time
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.applications.densenet import DenseNet121
from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

from dense_inception_concat import DenseNetInceptionConcat
from utils import Params

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data_dir", required=True,
                help="path to input dataset (i.e., directory of images)")
ap.add_argument("-c", "--ckpt_dir", required=True,
                help="path of check points (i.e., directory of check points)")
ap.add_argument("-r", "--restore_from", required=False,
                help="path of saved checkpoints (i.e., directory of check points)")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output accuracy/loss plot")
args = vars(ap.parse_args())



# initialize the number of epochs to train for, initial learning rate,

params = Params('params.json')

# config variables
EPOCHS = params.num_epochs
INIT_LR = params.learning_rate
BS = params.batch_size
IMAGE_DIMS = (params.image_size, params.image_size, 3)
seed      = 2019
results     = params.results_path
classifier_path = params.classifier_path
model_path = params.model_path


# Arguments
data_dir = args["data_dir"]
restore_from = args["restore_from"]
ckpt_dir = args["ckpt_dir"]

if restore_from is not None:
    use_pretrained = False
"""
"""
# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")
                    
"""
"""

if data_dir is None:
    data_dir = params.data_dir


train_dir = os.path.join(data_dir, "train")
valid_dir = os.path.join(data_dir, "test")
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMAGE_DIMS[1], IMAGE_DIMS[1]),
        batch_size=BS,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        valid_dir,
        target_size=(IMAGE_DIMS[1], IMAGE_DIMS[1]),
        batch_size=BS,
        class_mode='categorical')
# initialize the model
CLASSES = train_generator.num_classes
print(CLASSES)

if restore_from is not None:
    use_imagenet_weights = False
else:
    use_imagenet_weights = True

print("[INFO] compiling model...")

model = DenseNetInceptionConcat(num_labels=CLASSES, use_imagenet_weights=use_imagenet_weights).model


tensorBoard = TensorBoard(log_dir='logs/{}'.format(time.time()))
# checkpoint

checkpoint = ModelCheckpoint("output/checkpoints", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = checkpoint
if restore_from is not None:
    if os.path.isdir(restore_from):
        model.load_weights(restore_from)

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])


# train the network
print("[INFO] training network...")
H = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.n // train_generator.batch_size,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.n // validation_generator.batch_size,
        callbacks=[callbacks_list, tensorBoard])


# save the model to disk
print("[INFO] serializing network...")
model.save(os.path.join(ckpt_dir, "last.weights.h5"))


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])
# USAGE
# python train.py --dataset dataset --model pokedex.model --labelbin lb.pickle

# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import TensorBoard
import time
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from pyimagesearch.smallervggnet import SmallerVGGNet
from keras.applications.densenet import DenseNet121
from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os
from pyimagesearch.densenet_model import densenet121_model

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data_dir", required=True,
                help="path to input dataset (i.e., directory of images)")
ap.add_argument("-c", "--ckpt_dir", required=True,
                help="path of check points (i.e., directory of check points)")
ap.add_argument("-r", "--restore_from", required=False,
                help="path of saved checkpoints (i.e., directory of check points)")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 20
INIT_LR = 1e-2
BS = 20
CLASSES = 172
IMAGE_DIMS = (224, 224, 3)
use_pretrained = False

# Arguments
data_dir = args["data_dir"]
restore_from = args["restore_from"]
ckpt_dir = args["ckpt_dir"]

if restore_from is not None:
    use_pretrained = False
"""
"""
# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")
"""
"""

train_dir = os.path.join(data_dir, "train")
valid_dir = os.path.join(data_dir, "test")
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMAGE_DIMS[1], IMAGE_DIMS[1]),
        batch_size=BS,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        valid_dir,
        target_size=(IMAGE_DIMS[1], IMAGE_DIMS[1]),
        batch_size=BS,
        class_mode='categorical')
# initialize the model
CLASSES = train_generator.num_classes
print(CLASSES)
print("[INFO] compiling model...")
# model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
#                             depth=IMAGE_DIMS[2], classes=3)

#model = densenet121_model(img_rows=IMAGE_DIMS[0], img_cols=IMAGE_DIMS[1], color_type=IMAGE_DIMS[2], num_classes=CLASSES, use_pretrained=use_pretrained)
base_model = DenseNet121(include_top=False, weights='imagenet', input_tensor=Input(shape=(224,224,3)), input_shape=(224,224,3), pooling='avg')
x = Dense(CLASSES, activation='softmax')(base_model.get_layer('avg_pool').output)
model = Model(input=base_model.input, output=x)
tensorBoard = TensorBoard(log_dir='logs/{}'.format(time.time()))
# checkpoint

checkpoint = ModelCheckpoint(ckpt_dir, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = checkpoint
if restore_from is not None:
    if os.path.isdir(restore_from):
        model.load_weights(restore_from)

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])


# train the network
print("[INFO] training network...")
H = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.n // train_generator.batch_size,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.n // validation_generator.batch_size,
        callbacks=[callbacks_list, tensorBoard])


# save the model to disk
print("[INFO] serializing network...")
model.save(os.path.join(ckpt_dir, "last.weights.h5"))


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])
-----

"""