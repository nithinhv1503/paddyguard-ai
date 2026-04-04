"""
Paddy Leaf Disease Model Training
Uses MobileNetV2 transfer learning for better accuracy on the augmented dataset.
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

# Data augmentation for training (real-time augmentation on top of offline augmented data)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    fill_mode='reflect'
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    "dataset",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    "dataset",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

num_classes = len(train_data.class_indices)

# Print class mapping for verification
print("\n" + "="*50)
print("  CLASS MAPPING (alphabetical order)")
print("="*50)
for class_name, index in sorted(train_data.class_indices.items(), key=lambda x: x[1]):
    print(f"  Index {index}: {class_name}")
print("="*50)

# Save class indices to file for predict.py to use
with open("class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f, indent=2)
print("  Saved class_indices.json")

# Build model using MobileNetV2 transfer learning
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze base model layers initially
base_model.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

print("\n  Phase 1: Training classification head...")
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr]
)

# Phase 2: Fine-tune some layers of MobileNetV2
print("\n  Phase 2: Fine-tuning top layers...")
base_model.trainable = True

# Freeze all layers except the last 30
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=[early_stop, reduce_lr]
)

# Evaluate
val_loss, val_acc = model.evaluate(val_data)
print(f"\n  Final Validation Accuracy: {val_acc*100:.2f}%")
print(f"  Final Validation Loss:     {val_loss:.4f}")

# Save model
model.save("paddy_disease_model.h5")
print("\n  Model saved as paddy_disease_model.h5")
print("  Training complete!")