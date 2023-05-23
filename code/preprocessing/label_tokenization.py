from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.utils import pad_sequences
from keras.utils import to_categorical


# The `LabelTokenizer` class tokenizes target texts, pads them, and returns decoder input and target data as one-hot
# encoded vectors.
class LabelTokenizer:
    def __init__(self, padding_len):
        self.num_decoder_tokens = 50
        self.padding_len = padding_len
        self.tokenizer = Tokenizer(num_words=self.num_decoder_tokens)
        # Define the phrases to be predicted
        self.phrases = ['Stop navigation.', 'Excuse me.', 'I am sorry.', 'Thank you.', 'Good bye.', 'I love this game.',
                        'Nice to meet you.', 'You are welcome.', 'How are you?', 'Have a good time.']
        self.words = ['Begin', 'Choose', 'Connection', 'Navigation', 'Next', 'Previous', 'Start', 'Stop', 'Hello',
                      'Web']
        self.tokenizer.fit_on_texts(self.words + self.phrases)

    def get_label_tokens(self, target_texts):
        """
        This function takes in target texts, tokenizes them, pads them, and returns decoder input and target data.

        :param target_texts: The target texts are the texts that we want to generate labels for. In this function, we are
        converting these texts into sequences of tokens using a tokenizer. Then, we are padding these sequences to make them
        of equal length and converting them into one-hot encoded vectors. Finally, we are splitting these vectors
        :return: two lists: `decoder_input_data` and `decoder_target_data`. These lists contain the tokenized and padded
        sequences of the input target texts, where `decoder_input_data` contains all but the last token of each sequence,
        and `decoder_target_data` contains all but the first token of each sequence. The tokens are represented as one-hot
        encoded vectors of length `num_decoder_tokens
        """
        decoder_input_data = []
        decoder_target_data = []
        sequences = self.tokenizer.texts_to_sequences(target_texts)
        sequences_np = np.array(sequences, dtype=object)
        sequences_padded = pad_sequences(sequences_np, padding='post', truncating='post', maxlen=self.padding_len)
        for seq in sequences_padded:
            y = to_categorical(seq, self.num_decoder_tokens)
            decoder_input_data.append(y[:-1])
            decoder_target_data.append(y[1:])
        return decoder_input_data, decoder_target_data
