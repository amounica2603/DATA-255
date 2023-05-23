import argparse
import logging

from models.lipreading_lstm_model import LipreadingLSTMModel
from models.lipreading_rnn_attention_model import LipreadingRNNModel
from models.lipreading_self_attention_model import LipreadingSelfAttentionModel
from models.lipreading_stacked_gru_model import LipreadingGRUModel
from preprocessing.dataset_preprocess import LipReadingImageProcessor


def main(model_name='LSTM', num_epochs=100):
    data_processor = LipReadingImageProcessor('dataset', 'utility_models/shape_predictor_68_face_landmarks.dat')
    model = None
    if model_name == 'LSTM':
        data = data_processor.get_datasets_for_model()
        model = LipreadingLSTMModel(data, epochs=num_epochs)
    elif model_name == 'RNN':
        data = data_processor.get_datasets_for_model(padding_len=17)
        model = LipreadingRNNModel(data, epochs=num_epochs)
    elif model_name == 'GRU':
        data = data_processor.get_datasets_for_model()
        model = LipreadingGRUModel(data, epochs=num_epochs)
    elif model_name == 'SELF_ATTENTION':
        data = data_processor.get_datasets_for_model()
        model = LipreadingSelfAttentionModel(data, epochs=num_epochs)
    else:
        logging.error(f'No model available with the name: [{model_name}]\n '
                      f'Available models: [LSTM, RNN, GRU, SELF_ATTENTION]')
        return

    model.plot_loss_acc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        help='Name of the model that you like to run. '
                             'Available model names = [LSTM, GRU, RNN, SELF_ATTENTION]',
                        default='LSTM')
    parser.add_argument('--epochs',
                        help='Number of epochs to run in the model training. '
                             'The models are set for early stopping with a default of 100',
                        default=100)
    input_args = parser.parse_args()
    main(model_name=input_args.model_name, num_epochs=int(input_args.epochs))
