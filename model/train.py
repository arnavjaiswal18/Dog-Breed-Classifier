import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import build_model
import os  

# -----------------------------
# Configuration
# -----------------------------
TRAIN_DIR = "data/train_split"
VAL_DIR = "data/valid"   # create this if not yet

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

MODEL_SAVE_PATH = "model/weights/model.h5"

# -----------------------------
# Data Preprocessing
# -----------------------------
print("📂 Loading Data...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print(f"✅ Classes found: {train_data.num_classes}")

# -----------------------------
# Build Model
# -----------------------------
print("🧠 Building Model...")

model = build_model(num_classes=train_data.num_classes)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# Callbacks
# -----------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor='val_loss',
        save_best_only=True
    )
]

# -----------------------------
# Train Model
# -----------------------------
print("🚀 Training Started...")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("✅ Training Complete!")

# -----------------------------
# Save Final Model (optional)
# -----------------------------
if not os.path.exists("model/weights"):
    os.makedirs("model/weights")

model.save(MODEL_SAVE_PATH)

print(f"💾 Model saved at: {MODEL_SAVE_PATH}")