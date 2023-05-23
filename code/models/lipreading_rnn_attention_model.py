from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, SimpleRNN, Concatenate, Dot
from tensorflow.keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

time_steps_encoder = 16
num_encoder_tokens = 4096
num_decoder_tokens = 50

latent_dim = 512
time_steps_decoder = 5


# The LipreadingLSTMModel class defines a model that uses LSTM layers for encoding and decoding input sequences, and
# includes functions for training the model and plotting its loss and accuracy.
class LipreadingRNNModel:
    def __init__(self, data, epochs=100, initial_lr=0.01, validation_split=0.2):
        self.data = data
        self.epochs = epochs
        self.initial_lr = initial_lr
        self.validation_split = validation_split
        self.model = self.get_model()
        self.history = self.train_model()

    def get_model(self):
        """
        This function returns a model that uses LSTM layers for encoding and decoding input sequences.
        :return: a Keras model that takes in two inputs (encoder_inputs and decoder_inputs) and outputs decoder_outputs. The
        model consists of an encoder LSTM layer, a decoder LSTM layer, and a dense layer with softmax activation.
        """
        # Define the input shape
        encoder_inputs = Input(shape=(16, 4096))

        # Define the encoder RNN layer
        encoder_rnn = SimpleRNN(256, return_sequences=True, return_state=True, dropout=0.6, recurrent_dropout=0.6)
        # Get the output and state from the encoder
        encoder_outputs, state_h = encoder_rnn(encoder_inputs)
        print(state_h.shape)
        # Define the decoder RNN layer
        decoder_rnn = SimpleRNN(256, return_sequences=True, return_state=True, dropout=0.6, recurrent_dropout=0.6)
        # Define the attention layer
        attention_layer1 = Dense(16, activation='relu')
        attention_layer2 = Dense(16, activation='relu')
        attention_layer3 = Dense(16, activation='relu')

        # Define the output dense layer
        decoder_dense = Dense(50, activation='softmax')

        # Define the decoder input layer
        decoder_inputs = Input(shape=(16, 50))

        # Initialize the initial state with the encoder state
        initial_state = [state_h]

        # Get the decoder outputs and states from the decoder layer
        decoder_outputs, _ = decoder_rnn(decoder_inputs, initial_state=initial_state)
        print(decoder_outputs.shape)
        # Calculate the attention weights
        attention_weights = attention_layer1(decoder_outputs)
        attention_weights1 = attention_layer2(attention_weights)
        attention_weights2 = attention_layer3(attention_weights1)

        # Perform dot product between attention weights and encoder outputs
        context_vector = Dot(axes=1)([attention_weights2, encoder_outputs])

        # Concatenate the context vector and decoder output
        decoder_combined_context = Concatenate(axis=-1)([context_vector, decoder_outputs])

        # Pass the concatenated tensor through a dense layer
        output = decoder_dense(decoder_combined_context)

        # Define the model and compile
        model = Model([encoder_inputs, decoder_inputs], output)

        print(model.summary())

        return model

    def train_model(self):
        """
        This function trains a model using early stopping, model checkpointing, and learning rate reduction callbacks.
        :return: the training history of the model.
        """
        # Split the data into train and test sets
        train_encoder_input_data, test_encoder_input_data, train_decoder_input_data, test_decoder_input_data, train_decoder_target_data, test_decoder_target_data = train_test_split(
            self.data['encoder_ip'], self.data['decoder_ip'], self.data['decoder_trg'], test_size=0.2, random_state=42)

        # Split the train set into train and validation sets
        train_encoder_input_data, val_encoder_input_data, train_decoder_input_data, val_decoder_input_data, train_decoder_target_data, val_decoder_target_data = train_test_split(
            train_encoder_input_data, train_decoder_input_data, train_decoder_target_data, test_size=0.1,
            random_state=42)

        earlystopping = EarlyStopping(monitor='val_loss', patience=6, verbose=1, mode='min')
        checkpoint = ModelCheckpoint('saved_model_weights/lipreading_rnn_best_model.h5', monitor='val_loss', verbose=1,
                                     save_best_only=True, mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=0.0001)

        self.model.compile(metrics=['accuracy'], optimizer='adam', loss='categorical_crossentropy')

        history = self.model.fit([train_encoder_input_data, train_decoder_input_data], train_decoder_target_data,
                                 batch_size=64,
                                 epochs=100,
                                 validation_split=0.2,
                                 callbacks=[earlystopping, checkpoint, reduce_lr])

        # Evaluate the model on the validation set
        val_loss, val_acc = self.model.evaluate([val_encoder_input_data, val_decoder_input_data],
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
