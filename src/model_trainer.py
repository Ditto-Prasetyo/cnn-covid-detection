import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
from .utils import plot_training_history, show_confusion_matrix, compute_entropy_loss


class CovidClassifier:
    def __init__(self, dataset_path='dataset/', img_size=(224, 224), batch_size=32):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_labels = []
        self.model = None
        self.history = None
        self.train_generator = None
        self.val_generator = None

    def prepare_data(self):
        datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            rotation_range=15,
            zoom_range=0.2,
            horizontal_flip=True,
        )

        self.train_generator = datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )

        self.val_generator = datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )

        self.class_labels = list(self.train_generator.class_indices.keys())

    def build_model(self):
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(*self.img_size, 3))
        base_model.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(3, activation='softmax')(x)

        self.model = Model(inputs=base_model.input, outputs=predictions)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, epochs=20):
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.2, patience=3)
        ]

        self.history = self.model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=epochs,
            callbacks=callbacks
        )

    def evaluate(self):
        plot_training_history(self.history)

        Y_pred = self.model.predict(self.val_generator, verbose=1)
        y_pred = np.argmax(Y_pred, axis=1)
        y_true = self.val_generator.classes

        show_confusion_matrix(y_true, y_pred, self.class_labels)

        report = classification_report(y_true, y_pred, target_names=self.class_labels)
        print("\nClassification Report:\n", report)

        entropy = compute_entropy_loss(y_true, Y_pred, len(self.class_labels))
        print(f"\nManual Entropy Loss: {entropy:.4f}")

    def save(self, filename="model_covid_classifier.h5"):
        self.model.save(filename)
        print(f"Model saved to {filename}")
