
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
import numpy as np
import json

tokenizer = Tokenizer()
text_file_path = 'text'

def get_raw_data_from_file( path ):
    text = str()
    with open(path, "r") as fd:
        text += fd.read()
    return text

raw_text = get_raw_data_from_file( text_file_path )

corpus = raw_text.split( "\n\n" )
tokenizer.fit_on_texts(corpus)
total_words = len( tokenizer.word_index ) + 1

input_sequences = []

for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

sequence_lengths = list()
for x in input_sequences:
    sequence_lengths.append( len( x ) )
max_sequence_len = max( sequence_lengths )

input_sequences = np.array(pad_sequences(input_sequences,
                                         maxlen=max_sequence_len+1, padding='pre'))
x, y = input_sequences[:, :-1], input_sequences[:, -1]
y = keras.utils.to_categorical(y, num_classes=total_words)

np.save( 'data/x.npy', x)
np.save( 'data/y.npy', y)

config = { 'MAX_SEQUENCE_LEN' : max_sequence_len , 'TOTAL_WORDS' : total_words }
with open( 'data/config.json' , 'w' ) as file :
    json.dump( config , file )



