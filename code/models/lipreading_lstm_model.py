from tensorflow.keras.layers import Conv3D, MaxPooling3D, BatchNormalization, Input, Flatten, Dense, Reshape, \
    RepeatVector, LSTM, TimeDistributed
from tensorflow.keras import Sequential, Model
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

time_steps_encoder = 16
num_encoder_tokens = 4096
num_decoder_tokens = 50

latent_dim = 512
time_steps_decoder = 5


class LipreadingLSTMModel:
    def __init__(self, data, epochs=100, initial_lr=0.01, validation_split=0.2):
        self.data = data
        self.epochs = epochs
        self.initial_lr = initial_lr
        self.validation_split = validation_split
        self.model = self.get_model()
        self.history = self.train_model()

    def get_model(self):
        # encoder
        encoder_inputs = Input(shape=(time_steps_encoder, num_encoder_tokens), name="encoder_inputs")
        encoder = LSTM(latent_dim, return_state=True, return_sequences=True, name='endcoder_lstm')
        _, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]

        # decoder
        decoder_inputs = Input(shape=(time_steps_decoder, num_decoder_tokens), name="decoder_inputs")
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(num_decoder_tokens, activation='softmax', name='decoder_relu')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        print(model.summary())

        return model

    def train_model(self):
        earlystopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
        checkpoint = ModelCheckpoint('saved_model_weights/lipreading_lstm_best_model.h5', monitor='val_loss', verbose=1,
                                     save_best_only=True, mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=0.0001)

        opt = keras.optimizers.Adam(learning_rate=self.initial_lr)

        self.model.compile(metrics=['accuracy'], optimizer=opt, loss='categorical_crossentropy')

        history = self.model.fit([self.data['encoder_ip'], self.data['decoder_ip']], self.data['decoder_trg'],
                                 batch_size=64,
                                 epochs=self.epochs,
                                 validation_split=self.validation_split,
                                 callbacks=[earlystopping, checkpoint, reduce_lr])

        return history

    def plot_loss_acc(self):
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
