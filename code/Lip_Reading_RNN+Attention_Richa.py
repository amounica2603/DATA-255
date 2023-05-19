from google.colab import drive
from google.colab.patches import cv2_imshow
import shutil
from keras.applications.vgg16 import VGG16
from keras.models import Model
from tqdm import tqdm
import logging
import os
import cv2
import numpy as np
import dlib
from imutils import face_utils
from genericpath import isdir
import subprocess

from keras.utils.data_utils import pad_sequences
from keras.utils import to_categorical
import os
import random
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, SimpleRNN, LSTM, Concatenate, Dot, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


class LipReading:
    def __init__(self, base_dir_path='/content',
                 raw_dir_name='/content/gdrive/Shareddrives/DATA255_Deep_Learning/MIRACL-VC1_all_in_one',
                 processed_dir_name='/content/gdrive/Shareddrives/DATA255_Deep_Learning/Dataset/MIRACL_Processed_cnn_features/MIRACL_Processed_cnn_features',
                 delete_existing=False
                 ):
        self.DRIVE_DATASET_PATH = '/content/gdrive/Shareddrives/DATA255_Deep_Learning/Dataset'
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('/content/shape_predictor_68_face_landmarks.dat')
        # Define the phrases to be predicted
        self.phrases = ['Stop navigation.', 'Excuse me.', 'I am sorry.', 'Thank you.', 'Good bye.', 'I love this game.',
                        'Nice to meet you.', 'You are welcome.', 'How are you?', 'Have a good time.']
        self.words = ['Begin', 'Choose', 'Connection', 'Navigation', 'Next', 'Previous', 'Start', 'Stop', 'Hello',
                      'Web']

        self.num_decoder_tokens = 50
        self.base_dir_path = base_dir_path
        self.raw_dir_name = raw_dir_name
        self.delete_existing = delete_existing
        self.processed_dir_name = processed_dir_name
        #self.dataset_path = /path/to/dataset

        return

    def extract_mouth(self, gray_img):

        rects = self.detector(gray_img, 1)
        # print(rects)
        for (i, rect) in enumerate(rects):
            shape = self.predictor(gray_img, rect)
            shape = face_utils.shape_to_np(shape)
            for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                if name == "mouth":
                    # print(shape[i:j])
                    (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                    processed_frame = img[y:y + h, x:x + w]
                    return processed_frame

    # ### Data Preprocessing
    @staticmethod
    def get_dataset_metadata(self):
        # Uncomment to fetch from raw data
        # person_ids = sorted(os.listdir(dataset_path))
        # uttr_indexes = sorted(os.listdir(f'{dataset_path}/{person_ids[0]}/phrases/'))
        # instances = sorted(os.listdir(f'{dataset_path}/{person_ids[0]}/phrases/{uttr_indexes[0]}'))
        # return person_ids, uttr_indexes, instances

        return (['F01', 'F02', 'F04', 'F05', 'F06', 'F07', 'F08', 'F09', 'F10', 'F11',
                 'M01', 'M02', 'M04', 'M07', 'M08'],
                ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10'],
                ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10'])

    def preprocess_data(self):

        # specify the dataset path
        base_dir = self.base_dir_path
        dataset_path = os.path.join(base_dir, self.raw_dir_name)

        person_ids, uttr_indexes, instances = get_dataset_metadata(dataset_path)

        processed_dir = os.path.join(base_dir, self.processed_dir_name)

        if self.delete_existing:
            shutil.rmtree(processed_dir)

        if not os.path.isdir(processed_dir):
            os.mkdir(processed_dir)

        # iterate through all the instances in the phrases and words dataset
        for person_id in person_ids:
            print(f'processing {person_id}')
            if person_id == 'calib.txt':
                pass
            for uttr in ['words', 'phrases']:
                for uttr_index in uttr_indexes:
                    for instance_index in instances:
                        # load the color and depth frames for the instance
                        frames_path = os.path.join(dataset_path, f'{person_id}', f'{uttr}', f'{uttr_index}',
                                                   f'{instance_index}')
                        frames = sorted(os.listdir(frames_path))

                        # preprocess each frame
                        for i in range(len(frames) // 2):
                            img_path = os.path.join(frames_path, frames[i])
                        try:
                            # read the color and depth frames
                            img = cv2.imread(img_path)
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            dest_dir = os.path.join(processed_dir, f'{person_id}_{uttr}_{uttr_index}_{instance_index}')
                            if not os.path.isdir(dest_dir):
                                os.mkdir(dest_dir)
                            dest_fname = os.path.join(dest_dir, f'{i}.jpg')
                            processed_frame = extract_mouth(gray)
                            cv2.imwrite(dest_fname, processed_frame)
                            # print(f'processed img - {dest_fname}')
                            # cv2_imshow(processed_frame)
                            # raise Exception
                        except Exception as e:
                            print(f'Error while processing {img_path}')
                            print(f'Exception {e}')

        processed_zip = 'MIRACL_Processed.zip'
        if processed_zip in os.listdir(self.DRIVE_DATASET_PATH):
            print('Found the processed data zip, unpacking to local...')
            subprocess.check_output(f"unzip '{os.path.join(self.DRIVE_DATASET_PATH, processed_zip)}' -d . &> /dev/null",
                                    shell=True)
        else:
            print('Processed data %s does not exist, Preprocessing the raw data\n'%(processed_zip))
            preprocess_data(base_dir_path='/content', raw_dir_name='MIRACL-VC1_all_in_one',
                            processed_dir_name='MIRACL_Processed', delete_existing=True)

        print(f'Processed data available under: {processed_zip[:-4]}')

    @staticmethod
    def load_cnn_vgg_model(self):
        model = VGG16(weights="imagenet", include_top=True, input_shape=(224, 224, 3))
        out = model.layers[-2].output
        model_final = Model(inputs=model.input, outputs=out)
        return model_final

    @staticmethod
    def extract_instance_features(self, model, instance_path):
        image_list = os.listdir(instance_path)
        samples = np.round(np.linspace(
            0, len(image_list) - 1, 16))
        image_list = [image_list[int(sample)] for sample in samples]
        images = np.zeros((len(image_list), 224, 224, 3))
        for i in range(len(image_list)):
            img = cv2.imread(os.path.join(instance_path, image_list[i]))
            img = cv2.resize(img, (224, 224))
            images[i] = img
        images = np.array(images)
        fc_feats = model.predict(images, batch_size=16, verbose=0)
        img_feats = np.array(fc_feats)
        return img_feats

    def extract_save_features(self, preprocessed_dir, processed_dir):
        cnn_model = load_cnn_vgg_model()
        if processed_dir in os.listdir():
            logging.warn(f'Desired processed directory already exists: {processed_dir}')
        else:
            os.mkdir(processed_dir)
        for instance_path in tqdm(os.listdir(preprocessed_dir)):
            img_feature_stack = extract_instance_features(cnn_model,
                                                          os.path.join(preprocessed_dir, instance_path))
            np.save(os.path.join(processed_dir, instance_path),
                    img_feature_stack)

        processed_zip = 'MIRACL_Processed_cnn_features.zip'
        if processed_zip in os.listdir(self.DRIVE_DATASET_PATH):
            print('Found the processed data zip, unpacking to local...')
            subprocess.check_output(f"unzip '{os.path.join(DRIVE_DATASET_PATH, processed_zip)}' -d . &> /dev/null",
                                    shell=True)
        else:
            print('Extracting feature data')
            extract_save_features(preprocessed_dir='MIRACL_Processed',
                                  processed_dir='MIRACL_Processed_cnn_features')
        print(f'Extracted feature data available under: {processed_zip[:-4]}')

        return 0

    def get_datasets_for_model(self):
        # specify the preprocessed dataset path
        dataset_path = os.path.join(self.base_dir_path, self.raw_dir_name)

        person_ids, uttr_indexes, instances = self.get_dataset_metadata(dataset_path)

        tokenizer = Tokenizer(num_words=self.num_decoder_tokens)
        tokenizer.fit_on_texts(self.words + self.phrases)

        # Load the data into memory
        encoder_input_data = []
        decoder_input_data = []
        decoder_target_data = []
        target_texts = []
        for idx, (person_id, uttr, uttr_index, instance_index) in tqdm(
                enumerate([(person_id, uttr, uttr_index, instance_index)
                           for person_id in person_ids
                           for uttr in ('words', 'phrases')
                           for uttr_index in uttr_indexes
                           for instance_index in instances])):
            frames_path = os.path.join(self.base_dir_path, self.processed_dir_name,
                                       f'{person_id}_{uttr}_{uttr_index}_{instance_index}.npy')
            encoder_input_data.append(np.load(frames_path))
            if uttr == 'words':
                target_texts.append(f'<start> {self.words[uttr_indexes.index(uttr_index)]} <end>')
            else:
                target_texts.append(f'<start> {self.phrases[uttr_indexes.index(uttr_index)]} <end>')

        sequences = tokenizer.texts_to_sequences(target_texts)
        sequences_padded = np.array(sequences)
        sequences_padded = pad_sequences(sequences_padded, padding='post', truncating='post', maxlen=17)
        # print(len(sequences_padded))
        for seq in sequences_padded:
            y = to_categorical(seq, self.num_decoder_tokens)
            decoder_input_data.append(y[:-1])
            decoder_target_data.append(y[1:])
        # import pdb;pdb.set_trace()
        encoder_input_data = np.array(encoder_input_data)
        decoder_input_data = np.array(decoder_input_data)
        decoder_target_data = np.array(decoder_target_data)

        print('encoder_input_data shape:', encoder_input_data.shape)
        print('decoder_input_data shape:', decoder_input_data.shape)
        print('decoder_target_data shape:', decoder_target_data.shape)

        indices = np.random.permutation(encoder_input_data.shape[0])

        return {'encoder_ip': encoder_input_data[indices],
                'decoder_ip': decoder_input_data[indices],
                'decoder_trg': decoder_target_data[indices]}

    @staticmethod
    def return_history(data):

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
        # print(decoder_inputs.shape)
        # Initialize the initial state with the encoder state
        initial_state = [state_h]

        # Get the decoder outputs and states from the decoder layer
        decoder_outputs, _ = decoder_rnn(decoder_inputs, initial_state=initial_state)
        print(decoder_outputs.shape)
        # Calculate the attention weights
        attention_weights = attention_layer1(decoder_outputs)
        attention_weights1 = attention_layer2(attention_weights)
        attention_weights2 = attention_layer3(attention_weights1)

        # print(attention_weights.shape)
        # Perform dot product between attention weights and encoder outputs
        context_vector = Dot(axes=1)([attention_weights2, encoder_outputs])

        # Concatenate the context vector and decoder output
        decoder_combined_context = Concatenate(axis=-1)([context_vector, decoder_outputs])

        # Pass the concatenated tensor through a dense layer
        output = decoder_dense(decoder_combined_context)

        # Define the model and compile
        model = Model([encoder_inputs, decoder_inputs], output)
        model.compile(metrics=['accuracy'], optimizer='adam', loss='categorical_crossentropy')

        # from sklearn.model_selection import train_test_split

        # Split the data into train and test sets
        train_encoder_input_data, test_encoder_input_data, train_decoder_input_data, test_decoder_input_data, train_decoder_target_data, test_decoder_target_data = train_test_split(
            data['encoder_ip'], data['decoder_ip'], data['decoder_trg'], test_size=0.2, random_state=42)

        # Split the train set into train and validation sets
        train_encoder_input_data, val_encoder_input_data, train_decoder_input_data, val_decoder_input_data, train_decoder_target_data, val_decoder_target_data = train_test_split(
            train_encoder_input_data, train_decoder_input_data, train_decoder_target_data, test_size=0.1,
            random_state=42)

        # Train the model
        # import pdb;pdb.set_trace()

        #from tensorflow.keras.callbacks import EarlyStopping

        # Define early stopping callback
        #early_stopping = EarlyStopping(monitor='val_loss', patience=2)

        # Train the model with early stopping
        history = model.fit([train_encoder_input_data, train_decoder_input_data], train_decoder_target_data,
                            batch_size=64,
                            epochs=100,
                            validation_split=0.2)
                            #callbacks=[early_stopping])
        # Evaluate the model on the validation set
        val_loss, val_acc = model.evaluate([val_encoder_input_data, val_decoder_input_data], val_decoder_target_data)
        print("Validation loss:", val_loss)
        print("Validation accuracy:", val_acc)

        return history, model

    @staticmethod
    def plot_loss_acc(history, save=True):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
        # Plot the validation loss vs. train loss
        ax1.plot(history.history['loss'], label='Train Loss')
        ax1.plot(history.history['val_loss'], label='Val Loss', color='orange')
        ax1.legend()
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Plot the validation accuracy vs. train accuracy
        ax2.plot(history.history['accuracy'], label='Train Acc')
        ax2.plot(history.history['val_accuracy'], label='Val Acc')
        ax2.legend()
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        fig.tight_layout()
        if save:
          plt.savefig('/content/loss_accuracy.pdf')
        else:
          plt.show()


    @staticmethod
    def test_run(data, model):

        from sklearn.metrics import precision_score, f1_score, confusion_matrix
        # Split the data into train and test sets
        train_encoder_input_data, test_encoder_input_data, train_decoder_input_data, test_decoder_input_data, train_decoder_target_data, test_decoder_target_data = train_test_split(
            data['encoder_ip'], data['decoder_ip'], data['decoder_trg'], test_size=0.2, random_state=42)

        # Split the train set into train and validation sets
        train_encoder_input_data, val_encoder_input_data, train_decoder_input_data, val_decoder_input_data, train_decoder_target_data, val_decoder_target_data = train_test_split(
            train_encoder_input_data, train_decoder_input_data, train_decoder_target_data, test_size=0.1,
            random_state=42)

        # Verify the shapes of the data after splitting
        # print(train_encoder_input_data.shape)  # Should be the same as train_decoder_input_data
        # print(train_decoder_input_data.shape)
        # print(train_decoder_target_data.shape)  # Should match the number of samples

        # Make predictions on the test set
        predictions = model.predict([test_encoder_input_data, test_decoder_input_data])

        # Convert predictions to class labels
        predicted_labels = np.argmax(predictions, axis=-1)
        true_labels = np.argmax(test_decoder_target_data, axis=-1)

        # Calculate precision and F1-score
        precision = precision_score(true_labels.flatten(), predicted_labels.flatten(), average='macro')
        f1 = f1_score(true_labels.flatten(), predicted_labels.flatten(), average='macro')

        # Calculate confusion matrix
        confusion_mat = confusion_matrix(true_labels.flatten(), predicted_labels.flatten())

        # Print precision, F1-score, and confusion matrix
        # Evaluate the model on the validation set
        val_loss, val_acc = model.evaluate([val_encoder_input_data, val_decoder_input_data], val_decoder_target_data)
        print("Validation loss of RNN+ATTENTION model:", val_loss)
        print("Validation accuracy of RNN+ATTENTION model:", val_acc)
        print("Precision score of RNN+ATTENTION model: ", precision)
        print("F1-Score of RNN+ATTENTION model: ", f1)
        print("Confusion Matrix:\n", confusion_mat)


if __name__ == '__main__':
    lip_reading = LipReading()
    data = lip_reading.get_datasets_for_model()
    history, model = lip_reading.return_history(data)
    lip_reading.test_run(data, model)
    lip_reading.plot_loss_acc(history)


