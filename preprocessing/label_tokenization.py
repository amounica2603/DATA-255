from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.utils.data_utils import pad_sequences
from keras.utils import to_categorical

# Define the phrases to be predicted
phrases = ['Stop navigation.', 'Excuse me.', 'I am sorry.', 'Thank you.', 'Good bye.', 'I love this game.',
           'Nice to meet you.', 'You are welcome.', 'How are you?', 'Have a good time.']
words = ['Begin', 'Choose', 'Connection', 'Navigation', 'Next', 'Previous', 'Start', 'Stop', 'Hello', 'Web']

num_decoder_tokens = 50


class LabelTokenizer:
    def __init__(self):
        self.tokenizer = Tokenizer(num_words=num_decoder_tokens)
        self.tokenizer.fit_on_texts(words + phrases)

    def get_label_tokens(self, target_texts=None):
        if target_texts is None:
            target_texts = phrases + words
        decoder_input_data = []
        decoder_target_data = []
        sequences = self.tokenizer.texts_to_sequences(target_texts)
        sequences_padded = np.array(sequences)
        sequences_padded = pad_sequences(sequences_padded, padding='post', truncating='post', maxlen=6)
        for seq in sequences_padded:
            y = to_categorical(seq, num_decoder_tokens)
            decoder_input_data.append(y[:-1])
            decoder_target_data.append(y[1:])
        return decoder_input_data, decoder_target_data
