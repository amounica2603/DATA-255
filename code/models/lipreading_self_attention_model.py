from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import GRU, Input, Dense, Reshape, MultiHeadAttention
from tensorflow.keras.models import Model
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

time_steps_encoder = 16
num_encoder_tokens = 4096
num_decoder_tokens = 50

latent_dim = 512
time_steps_decoder = 5


# The LipreadingLSTMModel class defines a model that uses LSTM layers for encoding and decoding input sequences, and
# includes functions for training the model and plotting its loss and accuracy.
class LipreadingSelfAttentionModel:
    def __init__(self, data, epochs=100, initial_lr=0.01, validation_split=0.2):
        self.data = data
        self.epochs = epochs
        self.initial_lr = initial_lr
        self.validation_split = validation_split
        self.model = self.get_model()
        self.history = self.train_model()

    def get_model(self):
        encoder_inputs = Input(shape=(16, 4096))
        encoder_outputs = Reshape((-1, 4096))(encoder_inputs)

        # Decoder
        decoder_inputs = Input(shape=(6, 4096))
        decoder_outputs = Reshape((-1, 4096))(decoder_inputs)

        # Multi-head attention with encoder-decoder attention
        attention = MultiHeadAttention(num_heads=16, key_dim=128)
        decoder_outputs = attention(decoder_outputs, encoder_outputs)

        # Additional layers
        decoder_outputs = Dense(2048, activation='relu')(decoder_outputs)
        decoder_outputs = Dense(1024, activation='relu')(decoder_outputs)
        decoder_outputs = Dense(512, activation='relu')(decoder_outputs)

        # Concatenate encoder-decoder attention output and decoder inputs
        concat_inputs = tf.concat([decoder_outputs, decoder_inputs], axis=-1)

        # Dense layer for prediction
        dense = Dense(50, activation='softmax')
        decoder_outputs = dense(concat_inputs)

        # Define the model
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        model.summary()

        return model

    def train_model(self):
        earlystopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
        checkpoint = ModelCheckpoint('saved_model_weights/lipreading_self_attention_best_model.h5', monitor='val_loss',
                                     verbose=1, save_best_only=True, mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=0.0001)

        opt = keras.optimizers.Adam(learning_rate=0.01)

        self.model.compile(metrics=['accuracy'], optimizer=opt, loss='categorical_crossentropy')

        import numpy as np

        # Pad the decoder inputs
        padded_decoder_inputs = np.pad(self.data['decoder_ip'], ((0, 0), (0, 1), (0, 4046)), mode='constant')

        # Pad the decoder targets
        padded_decoder_targets = np.pad(self.data['decoder_trg'], ((0, 0), (0, 1), (0, 0)), mode='constant')

        # Split the data into train and test sets
        train_encoder_input_data, test_encoder_input_data, train_decoder_input_data, test_decoder_input_data, train_decoder_target_data, test_decoder_target_data = train_test_split(
            self.data['encoder_ip'], padded_decoder_inputs, padded_decoder_targets, test_size=0.2, random_state=42)

        # Further split the training data into train and validation sets
        train_encoder_input_data, val_encoder_input_data, train_decoder_input_data, val_decoder_input_data, train_decoder_target_data, val_decoder_target_data = train_test_split(
            train_encoder_input_data, train_decoder_input_data, train_decoder_target_data, test_size=0.2,
            random_state=42)

        history = self.model.fit([train_encoder_input_data, train_decoder_input_data], train_decoder_target_data,
                                 batch_size=64,
                                 epochs=self.epochs,
                                 validation_split=0.2,
                                 callbacks=[earlystopping, checkpoint, reduce_lr])

        # Evaluate the model on the validation set
        val_loss, val_acc = self.model.evaluate([val_encoder_input_data, val_decoder_input_data, val_decoder_input_data,
                                                 val_decoder_input_data],
                                                val_decoder_target_data)
        print("Validation loss:", val_loss)
        print("Validation accuracy:", val_acc)

        return history

    def plot_loss_acc(self):
        """
        This function plots the training and validation loss and accuracy of a neural network model.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
        # Plot the validation loss vs. train loss
        ax1.plot(self.history.history['loss'], label='Train Loss')
        ax1.plot(self.history.history['val_loss'], label='Val Loss')
        ax1.legend()
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Plot the validation accuracy vs. train accuracy
        ax2.plot(self.history.history['accuracy'], label='Train Acc')
        ax2.plot(self.history.history['val_accuracy'], label='Val Acc')
        ax2.legend()
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        plt.show()
