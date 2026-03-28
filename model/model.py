import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model


def build_model(num_classes=120):
    """
    Build a CNN model using MobileNetV2 (Transfer Learning)
    """

    # -----------------------------
    # Load Pretrained Base Model
    # -----------------------------
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # -----------------------------
    # Freeze Base Model Layers
    # -----------------------------
    for layer in base_model.layers:
        layer.trainable = False

    # -----------------------------
    # Add Custom Layers
    # -----------------------------
    x = base_model.output
    x = GlobalAveragePooling2D()(x)   # Convert (7,7,1280) → (1280)
    x = Dense(256, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)

    # -----------------------------
    # Final Model
    # -----------------------------
    model = Model(inputs=base_model.input, outputs=output)

    return model


# -----------------------------
# Run this file directly (for testing)
# -----------------------------
if __name__ == "__main__":
    print("🚀 Building Model...")
    model = build_model()
    model.summary()