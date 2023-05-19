import shutil
import os
import cv2
import numpy as np
import dlib
from imutils import face_utils
from genericpath import isdir
import subprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.models import Model
from tqdm import tqdm
import logging
from tensorflow.keras.utils import pad_sequences
from keras.utils import to_categorical
import random
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import GRU, Input, Dense, Reshape, Concatenate
from tensorflow.keras.models import Model
from sklearn.metrics import precision_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import logging

class Preprocessor:
    def __init__(self, base_dir_path, raw_dir_name, processed_dir_name):
        self.base_dir_path = base_dir_path
        self.raw_dir_name = raw_dir_name
        self.processed_dir_name = processed_dir_name
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    def extract_mouth(self, gray_img):
        rects = self.detector(gray_img, 1)
        for (i, rect) in enumerate(rects):
            shape = self.predictor(gray_img, rect)
            shape = face_utils.shape_to_np(shape)
            for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                if name == "mouth":
                    (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                    processed_frame = gray_img[y:y + h, x:x + w]
                    return processed_frame

    def get_dataset_metadata(self, dataset_path):
        return (['F01', 'F02', 'F04', 'F05', 'F06', 'F07', 'F08', 'F09', 'F10', 'F11', 
                 'M01', 'M02', 'M04', 'M07', 'M08'],
                ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10'],
                ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10'])

    def preprocess_data(self, delete_existing=False):
        # Specify the dataset path
        base_dir = self.base_dir_path
        dataset_path = os.path.join(base_dir, self.raw_dir_name)

        person_ids, uttr_indexes, instances = self.get_dataset_metadata(dataset_path)

        processed_dir = os.path.join(base_dir, self.processed_dir_name)

        if delete_existing:
            shutil.rmtree(processed_dir)

        if not os.path.isdir(processed_dir):
            os.mkdir(processed_dir)

        for person_id in person_ids:
            print(f'Processing {person_id}')
            if person_id == 'calib.txt':
                pass
            for uttr in ['words', 'phrases']:
                for uttr_index in uttr_indexes:
                    for instance_index in instances:
                        frames_path = os.path.join(dataset_path, f'{person_id}', f'{uttr}', f'{uttr_index}', f'{instance_index}')
                        frames = sorted(os.listdir(frames_path))

                        for i in range(len(frames)//2):
                            img_path = os.path.join(frames_path, frames[i])
                            try:
                                img = cv2.imread(img_path)
                                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                                processed_frame = self.extract_mouth(gray)
                                if processed_frame is not None:
                                    output_dir = os.path.join(processed_dir, person_id, uttr, uttr_index, instance_index)
                                    os.makedirs(output_dir, exist_ok=True)
                                    output_path = os.path.join(output_dir, frames[i])
                                    cv2.imwrite(output_path, processed_frame)
                            except Exception as e:
                                print(f'Error processing image: {img_path}')
                                print(e)
                                continue

class DatasetLoader:
    def __init__(self, base_dir_path, raw_dir_name, processed_dir_name):
        self.base_dir_path = base_dir_path
        self.raw_dir_name = raw_dir_name
        self.processed_dir_name = processed_dir_name
        self.num_decoder_tokens = 50
        self.phrases = ['Stop navigation.', 'Excuse me.', 'I am sorry.', 'Thank you.', 'Good bye.',
                        'I love this game.', 'Nice to meet you.', 'You are welcome.', 'How are you?',
                        'Have a good time.']
        self.words = ['Begin', 'Choose', 'Connection', 'Navigation', 'Next', 'Previous', 'Start', 'Stop',
                      'Hello', 'Web']
        self.encoder_input_data = []
        self.decoder_input_data = []
        self.decoder_target_data = []
        self.target_texts = []

    def get_datasets_for_model(self):
        # specify the preprocessed dataset path
        dataset_path = os.path.join(self.base_dir_path, self.raw_dir_name)

        person_ids, uttr_indexes, instances = self.get_dataset_metadata()

        tokenizer = Tokenizer(num_words=self.num_decoder_tokens)
        tokenizer.fit_on_texts(self.words + self.phrases)

        # Load the data into memory
        
        for idx, (person_id, uttr, uttr_index, instance_index) in tqdm(enumerate([(person_id, uttr, uttr_index, instance_index)
                                                                                    for person_id in person_ids
                                                                                    for uttr in ('words', 'phrases')
                                                                                    for uttr_index in uttr_indexes
                                                                                    for instance_index in instances])):
            frames_path = os.path.join(self.base_dir_path, self.processed_dir_name,
                                       f'{person_id}_{uttr}_{uttr_index}_{instance_index}.npy')
            self.encoder_input_data.append(np.load(frames_path))
            if uttr == 'words':
                self.target_texts.append(f'<start> {self.words[uttr_indexes.index(uttr_index)]} <end>')
            else:
                self.target_texts.append(f'<start> {self.phrases[uttr_indexes.index(uttr_index)]} <end>')

        sequences = tokenizer.texts_to_sequences(self.target_texts)
        sequences_padded = np.array(sequences)
        sequences_padded = pad_sequences(sequences_padded, padding='post', truncating='post', maxlen=6)
        for seq in sequences_padded:
            y = to_categorical(seq, self.num_decoder_tokens)
            self.decoder_input_data.append(y[:-1])
            self.decoder_target_data.append(y[1:])

        self.encoder_input_data = np.array(self.encoder_input_data)
        self.decoder_input_data = np.array(self.decoder_input_data)
        self.decoder_target_data = np.array(self.decoder_target_data)

        print('encoder_input_data shape:', self.encoder_input_data.shape)
        print('decoder_input_data shape:', self.decoder_input_data.shape)
        print('decoder_target_data shape:', self.decoder_target_data.shape)

        indices = np.random.permutation(self.encoder_input_data.shape[0])

        return {'encoder_ip': self.encoder_input_data[indices],
                'decoder_ip': self.decoder_input_data[indices],
                'decoder_trg': self.decoder_target_data[indices]}

    @staticmethod
    def get_dataset_metadata():
        # Implement your logic to retrieve the dataset metadata
        # and return the person_ids, uttr_indexes, and instances
        return (['F01', 'F02', 'F04', 'F05', 'F06', 'F07', 'F08', 'F09', 'F10', 'F11', 
                 'M01', 'M02', 'M04', 'M07', 'M08'],
                ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10'],
                ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10'])
    

class CNNFeatureExtractor:
    def __init__(self):
        self.model = self.load_cnn_vgg_model()

    @staticmethod
    def load_cnn_vgg_model():
        model = VGG16(weights="imagenet", include_top=True, input_shape=(224, 224, 3))
        out = model.layers[-2].output
        model_final = Model(inputs=model.input, outputs=out)
        return model_final

    @staticmethod
    def extract_instance_features(model, instance_path):
        print("Running on : {}".format(instance_path))
        image_list = os.listdir(instance_path)
        samples = np.round(np.linspace(0, len(image_list) - 1, 16))
        image_list = [image_list[int(sample)] for sample in samples]
        images = np.zeros((len(image_list), 224, 224, 3))
        for i in range(len(image_list)):
            img = cv2.imread(os.path.join(instance_path, image_list[i]))
            if img is None:
                continue
            height, width = img.shape[:2]
            print("Image size - Width: {} pixels, Height: {} pixels".format(width, height))
            img = cv2.resize(img, (224, 224))
            images[i] = img
        images = np.array(images)
        fc_feats = model.predict(images, batch_size=16, verbose=0)
        img_feats = np.array(fc_feats)
        return img_feats

    @staticmethod
    def extract_save_features(preprocessed_dir, processed_dir):
        cnn_model = CNNFeatureExtractor.load_cnn_vgg_model()
        if processed_dir in os.listdir():
            logging.warn(f'Desired processed directory already exists: {processed_dir}')
        else:
            os.mkdir(processed_dir)
        for instance_path in tqdm(os.listdir(preprocessed_dir)):
            img_feature_stack = CNNFeatureExtractor.extract_instance_features(cnn_model,
                                                                               os.path.join(preprocessed_dir,
                                                                                            instance_path))
            np.save(os.path.join(processed_dir, instance_path),
                    img_feature_stack)

    def process_data(self, preprocessed_dir='MIRACL_Processed', processed_dir='MIRACL_Processed_cnn_features'):
        processed_zip = 'MIRACL_Processed_cnn_features.zip'
        if processed_zip in os.listdir():
            print('Found the processed data zip, unpacking to local...')
            subprocess.check_output(f"unzip '{processed_zip}' -d . &> /dev/null", shell=True)
        else:
            print('Extracting feature data')
            self.extract_save_features(preprocessed_dir=preprocessed_dir, processed_dir=processed_dir)
        print(f'Extracted feature data available under: {processed_dir}')

    def run(self):
        self.process_data()



class ModelTrainer:
    def __init__(self, data, phrases, words):
        self.data = data
        self.phrases = phrases
        self.words = words

    def create_model(self):
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

        # Create the model
        model = self.create_model()
        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Fit the model on the training data with validation
        history = model.fit(
            [train_encoder_input_data, train_decoder_input_data, train_decoder_input_data, train_decoder_input_data],
            train_decoder_target_data,
            batch_size=64,
            epochs=100,
            validation_data=(
                [val_encoder_input_data, val_decoder_input_data, val_decoder_input_data, val_decoder_input_data],
                val_decoder_target_data))

        return model, history

    def plot_loss_acc(self, history):
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

        plt.show()

    def evaluate_model(self, model, encoder_input_data, decoder_input_data, decoder_target_data):
        # Evaluate the model on the test set
        test_loss, test_acc = model.evaluate(
            [encoder_input_data, decoder_input_data, decoder_input_data, decoder_input_data], decoder_target_data)
        print("Test loss:", test_loss)
        print("Test accuracy:", test_acc)

        # Make predictions on the test set
        predictions = model.predict(
            [encoder_input_data, decoder_input_data, decoder_input_data, decoder_input_data])
        predicted_labels = np.argmax(predictions, axis=-1)
        true_labels = np.argmax(decoder_target_data, axis=-1)

        # Calculate precision and F1-score
        precision = precision_score(true_labels.flatten(), predicted_labels.flatten(), average='macro')
        f1 = f1_score(true_labels.flatten(), predicted_labels.flatten(), average='macro')

        # Calculate confusion matrix
        confusion_mat = confusion_matrix(true_labels.flatten(), predicted_labels.flatten())

        # Print precision, F1-score, and confusion matrix
        print("Precision score: ", precision)
        print("F1-Score: ", f1)
        print("Confusion Matrix:\n", confusion_mat)

    def train_plot_evaluate(self):
        # Create and train the model
        model, history = self.train_model()

        # Plot the loss and accuracy curves
        self.plot_loss_acc(history)

        # Evaluate the model on the validation set
        val_loss, val_acc = model.evaluate(
            [self.val_encoder_input_data, self.val_decoder_input_data, self.val_decoder_input_data,
             self.val_decoder_input_data],
            self.val_decoder_target_data)
        print("Validation loss:", val_loss)
        print("Validation accuracy:", val_acc)

        # Evaluate the model on the test set
        self.evaluate_model(model, self.test_encoder_input_data, self.test_decoder_input_data,
                            self.test_decoder_target_data)



if __name__ == '__main__':
    base_dir_path = '/Users/mounicaayalasomayajula/Desktop/deep_learning/Dataset'
    raw_dir_name = 'MIRACL-VC1_all_in_one'
    processed_dir_name = 'MIRACL_Processed_cnn_features'
    processdata = Preprocessor(base_dir_path, raw_dir_name, processed_dir_name)
    # processdata.preprocess_data()
    # cnn_extractor = CNNFeatureExtractor()
    # cnn_extractor.run()

    dataset = DatasetLoader()
    data = {
        'encoder_ip': dataset.encoder_input_data,
        'decoder_ip': dataset.decoder_input_data,
        'decoder_trg': dataset.decoder_target_data
    }
    phrases = dataset.phrases
    words = dataset.words
    trainer = ModelTrainer(data, phrases, words)
    trainer.train_plot_evaluate()