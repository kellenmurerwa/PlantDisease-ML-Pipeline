import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import os

IMG_SIZE = (224, 224, 3)

def build_model(num_classes):
    base = tf.keras.applications.MobileNetV2(include_top=False, input_shape=IMG_SIZE, weights='imagenet')
    base.trainable = False  # freeze for initial training
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=base.input, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train(train_gen, val_gen, epochs=10, out_path="models/plantnet_v1.h5"):
    num_classes = train_gen.num_classes
    model = build_model(num_classes)
    checkpoint = callbacks.ModelCheckpoint(out_path, monitor='val_accuracy', save_best_only=True)
    stop = callbacks.EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True)
    history = model.fit(train_gen,
                        validation_data=val_gen,
                        epochs=epochs,
                        callbacks=[checkpoint, stop])
    return model, history

if __name__ == "__main__":
    import json
    from .preprocessing import get_generators
    train_gen, val_gen = get_generators(batch_size=32)
    model, history = train(train_gen, val_gen, epochs=15)
