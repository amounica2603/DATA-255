import os
import shutil
import logging
import cv2
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Model
from tqdm import tqdm

from preprocessing.label_tokenization import LabelTokenizer
from preprocessing.mouth_extract import ImageMouthExtractor


# The LipReadingImageProcessor class preprocesses lip reading data by extracting mouth images and extracting features
# using a VGG16 model.
class LipReadingImageProcessor:
    def __init__(self, dataset_base_path, shape_predictor_path):
        self.base_path = dataset_base_path
        self.raw_data_path = os.path.join(self.base_path, 'MIRACL-VC1_all_in_one')
        self.processed_data_path = os.path.join(self.base_path, 'MIRACL_Processed')
        self.processed_features_path = os.path.join(self.base_path, 'MIRACL_Processed_cnn_features')

        self.mouth_extractor = ImageMouthExtractor(shape_predictor_path)
        self.cnn_model = None

    def get_dataset_metadata(self):
        """
        This function retrieves metadata about the dataset, including the person IDs, utterance indices, and instance
        indices.
        """
        if os.path.isdir(self.raw_data_path):
            person_ids = sorted(os.listdir(self.raw_data_path))
            uttr_idxs = sorted(os.listdir(f'{self.raw_data_path}/{person_ids[0]}/phrases/'))
            instance_idxs = sorted(os.listdir(f'{self.raw_data_path}/{person_ids[0]}/phrases/{uttr_idxs[0]}'))

        else:
            person_ids = ['F01', 'F02', 'F04', 'F05', 'F06', 'F07', 'F08', 'F09', 'F10', 'F11',
                          'M01', 'M02', 'M04', 'M07', 'M08']
            uttr_idxs = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
            instance_idxs = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

        return person_ids, uttr_idxs, instance_idxs

    def generate_preprocessed_data(self, skip_if_processed=True):
        """
        This function preprocesses data by iterating through instances in a dataset, loading color and depth frames, and
        extracting mouth images from them.

        :param delete_existing: The delete_existing parameter is a boolean flag that determines whether to delete the
        existing processed data directory before preprocessing the data. If set to True, the existing directory will be
        deleted, defaults to True (optional)
        """
        if skip_if_processed:
            logging.info(f'Processed data is already available under location: {self.processed_data_path}')
            return
        person_ids, uttr_indexes, instances = self.get_dataset_metadata()

        if os.path.isdir(self.processed_data_path):
            shutil.rmtree(self.processed_data_path)

        if not os.path.isdir(self.processed_data_path):
            os.mkdir(self.processed_data_path)

        if not os.path.isdir(self.raw_data_path):
            logging.error(f'Raw data directory is missing: {self.raw_data_path}, '
                          f'make sure you executed the download_* scripts under utility_scripts/ ')
            return

        # iterate through all the instances in the phrases and words dataset
        for person_id in person_ids:
            print(f'processing {person_id}')
            if person_id == 'calib.txt':
                pass
            for uttr in ['words', 'phrases']:
                for uttr_index in uttr_indexes:
                    for instance_index in instances:
                        # load the color and depth frames for the instance
                        frames_path = os.path.join(self.raw_data_path, f'{person_id}', f'{uttr}', f'{uttr_index}',
                                                   f'{instance_index}')
                        frames = sorted(os.listdir(frames_path))

                        # preprocess each frame
                        for i in range(len(frames) // 2):
                            img_path = os.path.join(frames_path, frames[i])
                            try:
                                # read the color and depth frames
                                img = cv2.imread(img_path)
                                dest_dir = os.path.join(self.processed_data_path,
                                                        f'{person_id}_{uttr}_{uttr_index}_{instance_index}')
                                if not os.path.isdir(dest_dir):
                                    os.mkdir(dest_dir)
                                dest_fname = os.path.join(dest_dir, f'{i}.jpg')
                                processed_frame = self.mouth_extractor.extract_mouth(img)
                                cv2.imwrite(dest_fname, processed_frame)
                            except Exception as e:
                                print(f'Error while processing {img_path}')
                                print(f'Exception {e}')

    def _load_cnn_vgg_model(self):
        """
        This function loads a pre-trained VGG16 model with weights from ImageNet and removes the last layer to output the
        second to last layer.
        """
        model = VGG16(weights="imagenet", include_top=True, input_shape=(224, 224, 3))
        out = model.layers[-2].output
        model_final = Model(inputs=model.input, outputs=out)
        self.cnn_model = model_final

    def _extract_instance_features(self, instance_path):
        """
        This function extracts features from a set of images using a pre-trained CNN model and returns the features as an
        array.

        :param instance_path: The path to the directory containing the images of an instance
        :return: the image features extracted from a set of images in a given directory path. The image features are
        extracted using a pre-trained CNN model and returned as a numpy array.
        """
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
        fc_feats = self.cnn_model.predict(images, batch_size=16, verbose=0)
        img_feats = np.array(fc_feats)
        return img_feats

    def extract_save_img_features(self):
        """
        This function extracts and saves image features using a pre-trained CNN VGG model.
        """
        if not self.cnn_model:
            self._load_cnn_vgg_model()
        if self.processed_features_path in os.listdir():
            logging.warn(f'Desired processed directory already exists: {self.processed_features_path}')
        else:
            os.mkdir(self.processed_features_path)
        for instance_path in tqdm(os.listdir(self.processed_features_path)):
            img_feature_stack = self._extract_instance_features(os.path.join(self.processed_data_path, instance_path))
            np.save(os.path.join(self.processed_features_path, instance_path),
                    img_feature_stack)

    def get_datasets_for_model(self, padding_len=6):
        """
        This function loads preprocessed data into memory and returns the encoder input data, decoder input data, and
        decoder target data for a model.
        :return: A dictionary containing three keys: 'encoder_ip', 'decoder_ip', and 'decoder_trg', each with corresponding
        numpy arrays as their values.
        """
        # specify the preprocessed dataset path
        person_ids, uttr_indexes, instances = self.get_dataset_metadata()

        label_tokenizer = LabelTokenizer(padding_len)
        # Load the data into memory
        encoder_input_data = []
        target_texts = []
        for idx, (person_id, uttr, uttr_index, instance_index) in tqdm(
                enumerate([(person_id, uttr, uttr_index, instance_index)
                           for person_id in person_ids
                           for uttr in ('words', 'phrases')
                           for uttr_index in uttr_indexes
                           for instance_index in instances])):
            frames_path = os.path.join(self.processed_features_path,
                                       f'{person_id}_{uttr}_{uttr_index}_{instance_index}.npy')
            encoder_input_data.append(np.load(frames_path))
            if uttr == 'words':
                target_texts.append(f'<start> {label_tokenizer.words[uttr_indexes.index(uttr_index)]} <end>')
            else:
                target_texts.append(f'<start> {label_tokenizer.phrases[uttr_indexes.index(uttr_index)]} <end>')

        print(len(target_texts))
        decoder_input_data, decoder_target_data = label_tokenizer.get_label_tokens(target_texts)

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
