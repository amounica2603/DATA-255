import argparse
import logging
import os
from preprocessing.dataset_preprocess import LipReadingImageProcessor

# Change these only if you are using a different dataset location.
# Make sure to organize the dataset as per the project (in report) if you are using custom dataset and location.
DATASET_PATH = 'dataset'
SHAPE_PREDICTOR_MODEL_PATH = 'utility_models/shape_predictor_68_face_landmarks.dat'

logging.getLogger().setLevel('INFO')


# check for the data availability using the command run from DATA-255/code directory:
# python preprocess_data.py --check_data_availability True
def main(dataset_path, check_data_availability, preprocessed_data, features_data):
    data_preprocessor = LipReadingImageProcessor(dataset_base_path=dataset_path,
                                                 shape_predictor_path=SHAPE_PREDICTOR_MODEL_PATH)

    person_ids, phrase_word_ids, uttr_ids = data_preprocessor.get_dataset_metadata()

    logging.info(f'Person Ids: {person_ids}\n'
                 f'Phrases Ids: {phrase_word_ids}\n'
                 f'Words Ids: {phrase_word_ids}\n'
                 f'Utterance Ids for Phrases and Words: {uttr_ids}')

    if check_data_availability:
        if os.path.isdir(data_preprocessor.raw_data_path):
            logging.info(f'Raw data available under: {data_preprocessor.raw_data_path}')
        else:
            logging.warning(f'Raw data is not found under: {data_preprocessor.raw_data_path}')
        if os.path.isdir(data_preprocessor.processed_data_path):
            logging.info(f'Processed data available under: {data_preprocessor.processed_data_path}')
        else:
            logging.warning(f'Processed data is not found under: {data_preprocessor.processed_data_path}')
        if os.path.isdir(data_preprocessor.processed_features_path):
            logging.info(f'Features data available under: {data_preprocessor.processed_features_path}')
        else:
            logging.warning(f'Features data is not found under: {data_preprocessor.processed_features_path}')

    if preprocessed_data:
        data_preprocessor.generate_preprocessed_data(skip_if_processed=True)

    if features_data:
        data_preprocessor.extract_save_img_features()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', help='Dataset location. This is set default to <PROJECT_ROOT>/dataset/',
                        default=DATASET_PATH)
    parser.add_argument('--check_data_availability',
                        help='True if you need to check the availability of raw/processed/features data',
                        default=True)
    parser.add_argument('--preprocessed_data', help='True if you need to generate preprocessed data from raw data',
                        default=False)
    parser.add_argument('--features_data', help='True if you need to generate features data from preprocessed data',
                        default=False)
    input_args = parser.parse_args()

    main(input_args.dataset_path, input_args.check_data_availability,
         input_args.preprocessed_data, input_args.features_data)
