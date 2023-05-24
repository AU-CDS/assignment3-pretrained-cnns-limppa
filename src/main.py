##### IMPORT PACKAGES

import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

##### FUNCTIONS

# Plotting function
# This function plots the loss and accuracy curves.
def plot_history(H, epochs):
    plt.style.use("seaborn-colorblind")
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()

    plt.savefig(os.path.join("out", "learning_curves.jpg"))

##### MAIN

def main():
    # Load data
    test_dir = os.path.join("..", "images", "test")
    train_dir = os.path.join("..", "images", "train")
    val_dir = os.path.join("..", "images", "val")
    path = os.path.join("..", "images", "metadata")

    test_data = pd.read_json(os.path.join(path, "test_data.json"), lines=True)
    train_data = pd.read_json(os.path.join(path, "train_data.json"), lines=True)
    val_data = pd.read_json(os.path.join(path, "val_data.json"), lines=True)

    test_imgs = test_data["image_path"]
    train_imgs = train_data["image_path"]
    val_imgs = val_data["image_path"]

    y_test = test_data["class_label"]
    y_train = train_data["class_label"]
    y_val = val_data["class_label"]

    test = {'image_path': test_imgs, 'label': y_test}
    train = {'image_path': train_imgs, 'label': y_train}
    val = {'image_path': val_imgs, 'label': y_val}
    test_df = pd.DataFrame(test)
    train_df = pd.DataFrame(train)
    val_df = pd.DataFrame(val)

    sup_path = os.path.join("..", "..")

    test_df['image_path'] = test_df['image_path'].apply(lambda x: os.path.join(sup_path, "images", x))
    train_df['image_path'] = train_df['image_path'].apply(lambda x: os.path.join(sup_path, "images", x))
    val_df['image_path'] = val_df['image_path'].apply(lambda x: os.path.join(sup_path, "images", x))

    lb = LabelBinarizer()
    y_test = lb.fit_transform(y_test)
    y_train = lb.fit_transform(y_train)
    y_val = lb.fit_transform(y_val)

    labelNames = [
        'blouse', 'dhoti_pants', 'dupattas', 'gowns', 'kurta_men', 'leggings_and_salwars',
        'lehenga', 'mojaris_men', 'mojaris_women', 'nehru_jackets', 'palazzos', 'petticoats',
        'saree', 'sherwanis', 'women_kurta'  # alphabetical order
    ]

    # Create model
    model = VGG16(include_top=False, pooling='avg', input_shape=(224, 224, 3))

    for layer in model.layers:
        layer.trainable = False

    flat1 = Flatten()(model.layers[-1].output)
    bn = BatchNormalization()(flat1)
    class1 = Dense(256, activation='relu')(bn)
    class2 = Dense(128, activation='relu')(class1)
    output = Dense(15, activation='softmax')(class2)

    model = Model(inputs=model.inputs, outputs=output)

    lr_schedule = ExponentialDecay(initial_learning_rate=0.01, decay_steps=10000, decay_rate=0.9)
    sgd = SGD(learning_rate=lr_schedule)

    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    # Define flow
    IMG_SHAPE = 224
    batch_size = 128

    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,
        rotation_range=20,
        preprocessing_function=lambda x: tf.image.resize(x, (IMG_SHAPE, IMG_SHAPE))
    )

    img_iter_test = datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=test_dir,
        x_col="image_path",
        y_col="label",
        target_size=(IMG_SHAPE, IMG_SHAPE),
        batch_size=batch_size,
        class_mode="categorical"
    )

    img_iter_train = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=train_dir,
        x_col="image_path",
        y_col="label",
        target_size=(IMG_SHAPE, IMG_SHAPE),
        batch_size=batch_size,
        class_mode="categorical"
    )

    img_iter_val = datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=val_dir,
        x_col="image_path",
        y_col="label",
        target_size=(IMG_SHAPE, IMG_SHAPE),
        batch_size=batch_size,
        class_mode="categorical"
    )

    # Train model
    epochs = 10
    H = model.fit(img_iter_train, validation_data=img_iter_val, epochs=epochs)

    # Plot
    plot_history(H, epochs)

    # Make predictions
    predictions = model.predict(img_iter_test, batch_size=batch_size)

    # Prepare classification report
    report = classification_report(
        y_test.argmax(axis=1),
        predictions.argmax(axis=1),
        target_names=labelNames
    )

    outpath = os.path.join("..", "out", "report.txt")
    with open(outpath, 'w') as f:
        f.write(report)

if __name__ == "__main__":
    main()
