# Import libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from keras import layers
from keras import callbacks
from keras import optimizers
from keras import activations
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix

RANDOM_STATE = 43 # Initial state for random

translate = { # Dictionary
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "scoiattolo": "squirrel",
    "ragno": "spider"
}

###############################################################################################################################

# Get overview of dataset
# Provide the path to your dataset
DATA_PATH = 'C:/Users/Lenovo/OneDrive - student.vgu.edu.vn/Year_3/3_Machinelles_Lernen/Lab/Project/raw-img'

animals_dict = {'class': [], 'count': []}
for dir_name in os.listdir(DATA_PATH):
    src = os.path.join(DATA_PATH, dir_name) # (ex: source = cane folder's path)
    if dir_name in translate.keys(): # (ex: if cane in cane, cavallo, elefante,...)
        dst = os.path.join(DATA_PATH, translate[dir_name]) # (ex: destination = dog folder's path)
        os.rename(src, dst) # Rename (ex: cane folder => dog folder)
        animals_dict['class'].append(translate[dir_name]) # (ex: + dog)
        animals_dict['count'].append(len(os.listdir(dst))) # (ex: + nums of dog images)
    else: # never happen
        animals_dict['class'].append(dir_name)
        animals_dict['count'].append(len(os.listdir(src)))

animals_df = pd.DataFrame(animals_dict)
print(animals_dict)
print(animals_df)

###############################################################################################################################

# # Display 5 first original images
# def display_first_images(df, data_path, num_images=5):
#     # Create a subplot with 1 row and 'num_images' columns
#     fig, axs = plt.subplots(1, num_images, figsize=(15, 3))

#     for i in range(num_images):
#         class_name = df.loc[i, 'class']
#         file_name = os.listdir(os.path.join(data_path, class_name))[0]
#         image_path = os.path.join(data_path, class_name, file_name)

#         # Load and display the image
#         img = plt.imread(image_path)
#         axs[i].imshow(img)
#         axs[i].set_title(f'Class: {class_name}')
#         axs[i].axis('off')

#     plt.show()

# display_first_images(animals_df, DATA_PATH, num_images=5)

# ###############################################################################################################################

# # Pie chart for distribution display
# total_samples = sum(animals_df['count']) # Calculate total samples
# fig, ax = plt.subplots(figsize=(7, 6)) # Create a pie chart
# colors = plt.cm.tab10(range(len(animals_df))) # Use a colormap with enough distinct colors

# wedges, texts, autotexts = ax.pie(animals_df['count'],
#                                   labels=animals_df['class'],
#                                   autopct=lambda p: f'{p:.1f}%\n({int(p * total_samples / 100)})',
#                                   colors=colors,
#                                   textprops=dict(color="black"))

# centre_circle = plt.Circle((0, 0), 0.70, fc='white')
# fig = plt.gcf()
# fig.gca().add_artist(centre_circle) # Draw a white circle at the center (for aesthetics)
# plt.text(0, 0, f'Total\n{total_samples}', ha='center', va='center', fontsize=12, color='black') # total number of samples
# ax.axis('equal') # ensures that the pie is drawn as a circle.
# plt.title('Class distribution in the dataset', size=15) # Set the title

# for w in wedges: # Set the edge color to black and width to 2
#     w.set_edgecolor('black')
#     w.set_linewidth(2) 

# plt.show()

# ###############################################################################################################################

# # Load data
# def load_and_preprocess_data(data_path):
#     class_names = os.listdir(data_path) # (ex: class_names = [butterfly, cat,...])
#     num_classes = len(class_names) # num_classes = 10

#     df = {'filename': [], 'class': []}
#     for class_name in class_names:
#         class_path = os.path.join(data_path, class_name)
#         for filename in os.listdir(class_path):
#             df['filename'].append(os.path.join(class_path, filename))
#             df['class'].append(class_name)

#     data_df = pd.DataFrame(df)
    

#     train_df, test_df = train_test_split(data_df, test_size=0.2,
#                                         stratify=data_df['class'],
#                                         random_state=42)

#     return train_df, test_df, num_classes

# train_df, test_df, num_classes = load_and_preprocess_data(DATA_PATH)

# ###############################################################################################################################

# # Image data augmentation (preprocessing)
# datagen = ImageDataGenerator(
#     rescale=1./255,            # Rescale pixel values to the range [0, 1]
#     rotation_range=20,         # Random rotation of the image in the range [-20, 20] degrees
#     width_shift_range=0.2,     # Random horizontal shift in the range [-20%, 20%] of the image width
#     height_shift_range=0.2,    # Random vertical shift in the range [-20%, 20%] of the image height
#     shear_range=0.2,           # Shear intensity (shear angle in the range of [-20%, 20%])
#     zoom_range=0.2,            # Random zoom in the range [80%, 120%]
#     horizontal_flip=True,      # Randomly flip the image horizontally
#     fill_mode='nearest',       # Strategy for filling in newly created pixels (e.g., due to rotation or width/height shift)
#     validation_split=0.2,
# )

# ###############################################################################################################################

# # Data generators (create tensors: 2D matrices)
# BATCH_SIZE = 32
# IMG_SIZE = (224, 224)

# train_generator = datagen.flow_from_dataframe(
#     dataframe=train_df,
#     x_col='filename',
#     y_col='class',
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     shuffle=True,
#     seed=RANDOM_STATE,
#     subset='training'
# )

# val_generator = datagen.flow_from_dataframe(
#     dataframe=train_df,
#     x_col='filename',
#     y_col='class',
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     shuffle=True,
#     seed=RANDOM_STATE,
#     subset='validation'
# )

# test_generator = datagen.flow_from_dataframe(
#     dataframe=test_df,
#     x_col='filename',
#     y_col='class',
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     shuffle=False,
# )

# ###############################################################################################################################

# # Display some sample images from the training dataset
# def display_images(generator, num_images=5):
#     images, labels = next(generator) # images = vector of img; labels = vector of class
#     class_labels = list(generator.class_indices.keys()) # Get the class labels

#     plt.figure(figsize=(15, 6)) # Plot the images
#     for i in range(num_images):
#         plt.subplot(1, num_images, i + 1)
#         plt.imshow(images[i]) # vector => image
#         plt.title(f"Class: {class_labels[np.argmax(labels[i])]}")
#         plt.axis("off")

#     plt.show()

# display_images(train_generator)

# ###############################################################################################################################

# # Calculate class weights
# classes = np.unique(train_df['class']) # Get unique classes from the training data
# class_dirs = os.listdir(DATA_PATH)

# class_weights = {}
# total_samples = sum([len(os.listdir(os.path.join(
#     DATA_PATH, dir_label))) for dir_label in class_dirs])

# for idx, label in enumerate(class_dirs):
#     class_weights[idx] = total_samples / (2 * len(os.listdir(os.path.join(
#                                                          DATA_PATH, label))))

# print(class_weights)

# ###############################################################################################################################

# # Define Res Unit
# INPUT_SHAPE = (*IMG_SIZE, 3)

# class ResidualUnit(layers.Layer):
#     def __init__(self, filters, strides=1, activation="relu", **kwargs):
#         super().__init__(**kwargs)
#         self.activation = activations.get(activation)
#         self.main_layers = [
#             layers.Conv2D(filters, 3, strides=strides,
#                                 padding="same", use_bias=False),
#             layers.BatchNormalization(),
#             self.activation,
#             layers.Conv2D(filters, 3, strides=1,
#                                 padding="same", use_bias=False),
#             layers.BatchNormalization()
#         ]
#         self.skip_layers = []
#         if strides > 1:
#             self.skip_layers = [
#                 layers.Conv2D(filters, 1, strides=strides,
#                                     padding="same", use_bias=False),
#                 layers.BatchNormalization()
#             ]

#     def call(self, inputs):
#         Z = inputs
#         for layer in self.main_layers:
#             Z = layer(Z)
#         skip_Z = inputs
#         for layer in self.skip_layers:
#             skip_Z = layer(skip_Z)
#         return self.activation(Z + skip_Z)

# ###############################################################################################################################

# # Buil model
# input_layer = layers.Input(shape=(224, 224, 3)) # Input layer

# # Initial convolutional layer with batch normalization, ReLU activation, and max pooling
# x = layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False)(input_layer)
# x = layers.BatchNormalization()(x)
# x = layers.Activation("relu")(x)
# x = layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)

# # Residual units with increasing filters and varying strides
# prev_filters = 64
# count = 0
# for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
#     strides = 1 if filters == prev_filters else 2
#     x = ResidualUnit(filters, strides=strides)(x)
#     prev_filters = filters

# x = layers.GlobalAvgPool2D()(x) # Global average pooling
# x = layers.Flatten()(x) # Flatten
# output_layer = layers.Dense(10, activation="softmax")(x) # Final dense layer for classification

# model = models.Model(inputs=input_layer, outputs=output_layer) # Create the model
# model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()

# ###############################################################################################################################

# # Define call backs
# EPOCHS = 70

# stop_callback = tf.keras.callbacks.EarlyStopping(
#     monitor='val_loss',
#     min_delta=1e-5,
#     patience=4,
#     restore_best_weights=True
# )

# lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)

# ###############################################################################################################################

# # Train model
# history = model.fit(
#     train_generator,
#     epochs=EPOCHS,
#     validation_data=val_generator,
#     class_weight=class_weights,
#     callbacks=[stop_callback, lr_scheduler]
# )

# ###############################################################################################################################

# # Evaluation with test set
# results = model.evaluate(test_generator, verbose=0)

# print("    Test Loss: {:.5f}".format(results[0]))
# print("Test Accuracy: {:.2f}%".format(results[1] * 100))

# model.save(os.path.join('models','imageclassifier.h5')) # Save model

# ###############################################################################################################################

# # Acc and loss curves
# accuracy = history.history['accuracy']
# val_accuracy = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(len(accuracy))

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ax1.plot(epochs, accuracy, 'b', label='Training accuracy')
# ax1.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
# ax1.set_title('Training and validation accuracy')
# ax1.set_xlabel('Epochs')
# ax1.set_ylabel('Accuracy')
# ax1.legend()

# ax2.plot(epochs, loss, 'b', label='Training loss')
# ax2.plot(epochs, val_loss, 'r', label='Validation loss')
# ax2.set_title('Training and validation loss')
# ax2.set_xlabel('Epochs')
# ax2.set_ylabel('Loss')
# ax2.legend()

# fig.suptitle('Training and validation metrics', fontsize=16)
# plt.show()

# ###############################################################################################################################

# # Predict the label of the test_images
# pred = model.predict(test_generator)
# pred = np.argmax(pred,axis=1)

# # Map the label
# labels = (train_generator.class_indices)
# labels = dict((v,k) for k,v in labels.items())
# pred = [labels[k] for k in pred]

# y_test = list(test_df['class'])
# print(classification_report(y_test, pred))

# ###############################################################################################################################

# # Confusion matrix
# cm = confusion_matrix(y_test, pred)

# sns.heatmap(cm,
#             annot=True,
#             fmt='g',
#             xticklabels=list(labels.values()),
#             yticklabels=list(labels.values()))
# plt.ylabel('Classification',fontsize=13)
# plt.xlabel('Actual',fontsize=13)
# plt.title('Confusion Matrix',fontsize=17)
# plt.show()
