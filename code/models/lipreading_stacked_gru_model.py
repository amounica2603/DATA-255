from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import GRU, Input, Dense, Reshape, Concatenate
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
class LipreadingGRUModel:
    def __init__(self, data, epochs=100, initial_lr=0.01, validation_split=0.2):
        self.data = data
        self.epochs = epochs
        self.initial_lr = initial_lr
        self.validation_split = validation_split
        self.model = self.get_model()
        self.history = self.train_model()

    def get_model(self):
        encoder_inputs = Input(shape=(16, 4096))
        # Create three layers of GRU
        encoder_gru = GRU(256, return_sequences=True, return_state=True, name='encoder_gru')
        encoder_gru_2 = GRU(256, return_sequences=True, return_state=True, name='encoder_gru_2')
        encoder_gru_3 = GRU(256, return_sequences=True, return_state=True, name='encoder_gru_3')

        # Forward pass for the first input sequence
        _, encoder_state_1 = encoder_gru(encoder_inputs)

        # Reshape the output state to match the expected input shape of encoder_gru_2
        encoder_state_1_reshaped = Reshape((-1, 256))(encoder_state_1)

        _, encoder_state_2 = encoder_gru_2(encoder_state_1_reshaped)

        # Reshape the output state to match the expected input shape of encoder_gru_3
        encoder_state_2_reshaped = Reshape((-1, 256))(encoder_state_2)

        _, encoder_state_3 = encoder_gru_3(encoder_state_2_reshaped)

        # Rest of the code remains the same...

        # Forward pass for the second input sequence
        decoder_inputs_1 = Input(shape=(5, 50))
        decoder_gru = GRU(256, return_sequences=True, name='decoder_gru')
        decoder_outputs_1 = decoder_gru(decoder_inputs_1, initial_state=encoder_state_3)

        # Forward pass for the third input sequence
        decoder_inputs_2 = Input(shape=(5, 50))
        decoder_gru_2 = GRU(256, return_sequences=True, name='decoder_gru_2')
        decoder_outputs_2 = decoder_gru_2(decoder_inputs_2, initial_state=encoder_state_3)

        # Forward pass for the fourth input sequence
        decoder_inputs_3 = Input(shape=(5, 50))
        decoder_gru_3 = GRU(256, return_sequences=True, name='decoder_gru_3')
        decoder_outputs_3 = decoder_gru_3(decoder_inputs_3, initial_state=encoder_state_3)

        # Concatenate the output of the three decoders
        decoder_outputs = [decoder_outputs_1, decoder_outputs_2, decoder_outputs_3]
        non_empty_outputs = [x for x in decoder_outputs if x is not None]
        if len(non_empty_outputs) > 1:
            decoder_outputs = Concatenate()(non_empty_outputs)
        else:
            decoder_outputs = non_empty_outputs[0]

        # Define the dense output layer
        dense = Dense(50, activation='softmax', name='output')
        decoder_outputs = dense(decoder_outputs)

        # Define the model
        model = Model([encoder_inputs, decoder_inputs_1, decoder_inputs_2, decoder_inputs_3], decoder_outputs)

        model.summary()

        return model

    def train_model(self):
        # Define the input data
        encoder_ip = self.data['encoder_ip']
        decoder_ip = self.data['decoder_ip']
        decoder_trg = self.data['decoder_trg']

        # Split the data into train and test sets
        train_encoder_input_data, test_encoder_input_data, train_decoder_input_data, test_decoder_input_data, train_decoder_target_data, test_decoder_target_data = train_test_split(
            encoder_ip, decoder_ip, decoder_trg, test_size=0.2, random_state=42)

        # Further split the training data into train and validation sets
        train_encoder_input_data, val_encoder_input_data, train_decoder_input_data, val_decoder_input_data, train_decoder_target_data, val_decoder_target_data = train_test_split(
            train_encoder_input_data, train_decoder_input_data, train_decoder_target_data, test_size=0.2,
            random_state=42)

        # Compile the model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        earlystopping = EarlyStopping(monitor='val_loss', patience=6, verbose=1, mode='min')
        checkpoint = ModelCheckpoint('saved_model_weights/lipreading_gru_best_model.h5', monitor='val_loss', verbose=1,
                                     save_best_only=True, mode='min')

        # Fit the model on the training data with validation
        history = self.model.fit(
            [train_encoder_input_data, train_decoder_input_data, train_decoder_input_data, train_decoder_input_data],
            train_decoder_target_data,
            batch_size=64,
            epochs=self.epochs,
            validation_data=(
                [val_encoder_input_data, val_decoder_input_data, val_decoder_input_data, val_decoder_input_data],
                val_decoder_target_data),
            callbacks=[earlystopping, checkpoint]
        )

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
