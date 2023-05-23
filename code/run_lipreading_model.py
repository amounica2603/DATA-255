import argparse
import logging

from models.lipreading_lstm_model import LipreadingLSTMModel
from models.lipreading_rnn_attention_model import LipreadingRNNModel
from preprocessing.dataset_preprocess import LipReadingImageProcessor


def main(model_name='LSTM'):
    data_processor = LipReadingImageProcessor('dataset', 'utility_models/shape_predictor_68_face_landmarks.dat')
    data = data_processor.get_datasets_for_model()

    if model_name == 'LSTM':
        model = LipreadingLSTMModel(data)
    elif model_name == 'RNN':
        model = LipreadingRNNModel(data)
    elif model_name == 'GRU':
        model = LipreadingLSTMModel(data)
    elif model_name == 'SELF_ATTENTION':
        model = LipreadingLSTMModel(data)
    else:
        logging.error(f'No model available with the name: [{model_name}]\n '
                      f'Available models: [LSTM, RNN, GRU, SELF_ATTENTION]')

    model.plot_loss_acc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        help='Name of the model that you like to run. '
                             'Available model names = [LSTM, GRU, RNN, SELF_ATTENTION]',
                        default='LSTM')
    input_args = parser.parse_args()
    main(model_name=input_args.model_name)
