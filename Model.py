

from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import time
import tensorflow as tf
import winsound
import json

class Generator (object) :

    def __init__( self ):

        self.__test_X = None
        self.__test_Y = None

        dropout_rate = 0.3
        kernel_init = 'glorot_uniform'
        activation_func = keras.activations.relu

        self.__batch_size = 5

        with open('data/config.json') as f:
            self.__data = json.load(f)

        self.__tokenizer = Tokenizer()
        text_file_path = 'text'
        def get_raw_data_from_file(path):
            text = str()
            with open(path, "r") as fd:
                text += fd.read()
            return text
        raw_text = get_raw_data_from_file(text_file_path)
        corpus = raw_text.split("\n\n")
        self.__tokenizer.fit_on_texts(corpus)

        self.__SCHEMA = [

            Embedding( self.__data[ 'TOTAL_WORDS' ], 10, input_length=self.__data[ 'MAX_SEQUENCE_LEN']),
            LSTM( 32 ) ,
            Dropout(dropout_rate),
            Dense( 32 , activation=activation_func ) ,
            Dropout(dropout_rate),
            Dense( self.__data[ 'TOTAL_WORDS' ], activation=tf.nn.softmax )

        ]

        self.model = keras.Sequential(self.__SCHEMA)
        self.model.compile(
            optimizer=keras.optimizers.Adam() ,
            loss=keras.losses.categorical_crossentropy ,
            metrics=[ 'accuracy' ]
        )

    def validation_data (self, test_X=None , test_Y=None ) :
        self.__test_X = test_X
        self.__test_Y = test_Y

    def batch_size(self, batch_size):
        self.__batch_size = batch_size

    def fit(self, X, Y , number_of_epochs , plotTrainingHistory=True ):
        if plotTrainingHistory:
            tensorboard = TensorBoard('logs/{}'.format(time.time()))
            self.model.fit(
                X,
                Y,
                batch_size=self.__batch_size,
                epochs=number_of_epochs,
                callbacks=[ tensorboard  , self.CallBack() ]
            )
            self.model.summary()
        else:
            self.model.fit(
                X,
                Y,
                batch_size=self.__batch_size,
                epochs=number_of_epochs,
                callbacks=[ self.CallBack()]
            )
            self.model.summary()

    def predict(self , seed_text , seed=10 ):

        for i in range( seed ):

            token_list = self.__tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=
            self.__data['MAX_SEQUENCE_LEN'], padding='pre')
            predicted = self.model.predict_classes(token_list, verbose=0 )
            output_word = ""
            for word, index in self.__tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            seed_text += " " + output_word

        return seed_text

    def save_model( self , filepath  ):
        self.model.save(filepath)

    def load_model(self , file_path ):
        self.model = None
        self.model = keras.models.load_model(file_path)

    class CallBack( keras.callbacks.Callback ):

        def on_epoch_end(self, epoch, logs=None):
            winsound.PlaySound('C:/Users/Equip/Desktop/Shubham\'s Stuff/beep.wav', winsound.SND_FILENAME)

def predict(self , seed_text , seed=10 ):

    for i in range( seed ):

        token_list = self.__tokenizer.texts_to_sequences([seed_text])[0]
        print( token_list )
        token_list = pad_sequences([token_list], maxlen=
        self.__data['MAX_SEQUENCE_LEN'], padding='pre')
        predicted = self.model.predict_classes(token_list, verbose=0 )
        output_word = ""
        for word, index in self.__tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word

    return seed_text