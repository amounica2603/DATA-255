import os
import shutil
import logging
import cv2
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Model
from tqdm import tqdm

from preprocessing.mouth_extract import ImageMouthExtractor


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
        model = VGG16(weights="imagenet", include_top=True, input_shape=(224, 224, 3))
        out = model.layers[-2].output
        model_final = Model(inputs=model.input, outputs=out)
        self.cnn_model = model_final

    def _extract_instance_features(self, instance_path):
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
