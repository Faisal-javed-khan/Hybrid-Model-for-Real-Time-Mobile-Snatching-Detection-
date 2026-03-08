"""
TD_CNN-LSTM Model for Mobile Snatching Detection
Paper: "Hybrid Model for Real-Time Mobile Snatching Detection in Video Surveillance
        Using Time-Distributed CNN and Attention-Based LSTM"
Authors: Faisal Khan, Irshad Ahmad, Muhammad Zubair, Yasir Saleem Afridi
DOI: 10.21015/vtse.v14i1.2279
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, TimeDistributed, Conv2D, BatchNormalization,
    MaxPooling2D, Dropout, Flatten, LSTM, Dense, Layer
)
import tensorflow.keras.backend as K


class AttentionLayer(Layer):
    """
    Custom Attention Layer for LSTM outputs.
    Computes context vector as a weighted sum of hidden states.

    Attention weights: alpha_t = v^T * tanh(W_a * [x_t, h_i])
    Normalized via softmax, then context: C_t = sum(alpha_ti * h_i)
    """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[1], 1),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # Compute attention scores
        e = K.tanh(K.dot(x, self.W) + self.b)
        e = K.squeeze(e, axis=-1)

        # Normalize with softmax
        alpha = K.softmax(e)
        alpha = K.expand_dims(alpha, axis=-1)

        # Compute context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def build_td_cnn_lstm(
    num_frames=10,
    frame_height=240,
    frame_width=240,
    channels=3,
    lstm_units=10,
    dense_units=128,
    dropout_rate=0.7,
    num_classes=1
):
    """
    Build the TD_CNN-LSTM classifier model.

    Architecture:
        Input → TimeDistributed CNN (3 blocks) → LSTM → Attention → Dense → Output

    Args:
        num_frames (int): Number of frames per video sequence. Default: 10
        frame_height (int): Height of each frame. Default: 240
        frame_width (int): Width of each frame. Default: 240
        channels (int): Number of color channels (RGB=3). Default: 3
        lstm_units (int): Number of LSTM units. Default: 10
        dense_units (int): Units in the dense layer. Default: 128
        dropout_rate (float): Dropout rate for regularization. Default: 0.7
        num_classes (int): 1 for binary classification (sigmoid). Default: 1

    Returns:
        keras.Model: Compiled TD_CNN-LSTM model
    """
    inputs = Input(shape=(num_frames, frame_height, frame_width, channels),
                   name='video_input')

    # ─── Block 1: TimeDistributed CNN ────────────────────────────────────────
    # Conv Block 1 — 16 filters
    x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'),
                        name='conv1')(inputs)
    x = TimeDistributed(BatchNormalization(), name='bn1')(x)
    x = TimeDistributed(MaxPooling2D((2, 2)), name='pool1')(x)
    x = TimeDistributed(Dropout(0.25), name='drop1')(x)

    # Conv Block 2 — 32 filters
    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'),
                        name='conv2')(x)
    x = TimeDistributed(BatchNormalization(), name='bn2')(x)
    x = TimeDistributed(MaxPooling2D((2, 2)), name='pool2')(x)
    x = TimeDistributed(Dropout(0.25), name='drop2')(x)

    # Conv Block 3 — 64 filters
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'),
                        name='conv3')(x)
    x = TimeDistributed(BatchNormalization(), name='bn3')(x)
    x = TimeDistributed(MaxPooling2D((2, 2)), name='pool3')(x)
    x = TimeDistributed(Dropout(0.25), name='drop3')(x)

    # Flatten spatial features for each frame
    x = TimeDistributed(Flatten(), name='flatten')(x)

    # ─── Block 2: LSTM ───────────────────────────────────────────────────────
    x = LSTM(lstm_units, return_sequences=True, name='lstm')(x)

    # ─── Block 3: Attention Mechanism ────────────────────────────────────────
    x = AttentionLayer(name='attention')(x)

    # ─── Block 4: Classification Head ────────────────────────────────────────
    x = Dense(dense_units, activation='relu', name='dense')(x)
    x = Dropout(dropout_rate, name='dropout')(x)

    activation = 'sigmoid' if num_classes == 1 else 'softmax'
    outputs = Dense(num_classes, activation=activation, name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='TD_CNN_LSTM')
    return model


def compile_model(model, learning_rate=0.0001):
    """
    Compile the model with Adam optimizer.

    Args:
        model: Keras model to compile
        learning_rate (float): Learning rate for Adam optimizer. Default: 0.0001

    Returns:
        Compiled model
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


if __name__ == '__main__':
    model = build_td_cnn_lstm()
    model = compile_model(model)
    model.summary()
