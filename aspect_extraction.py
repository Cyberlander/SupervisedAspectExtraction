import xml.etree.ElementTree as ET
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import numpy as np
from keras import layers
from keras.models import Sequential
from keras.layers.pooling import MaxPooling1D
from keras.optimizers import Adam, RMSprop, SGD
from keras.layers.core import Activation
from keras.regularizers import l2




def get_sentences( tree ):
    corpus = tree.getroot()
    sentences = []
    sent = corpus.findall(".//sentence")
    for s in sent:
        sentences.append( s.find( 'text' ).text )
    return sentences

def process_sentences( sentences ):
    tokenizer = Tokenizer( num_words=20000 )
    tokenizer.fit_on_texts( sentences )
    sentences_indexed = tokenizer.texts_to_sequences( sentences )
    sentences_padded = sequence.pad_sequences( sentences_indexed, maxlen=MAX_SEQ_LENGTH, padding='post' )
    return sentences_padded


def get_output( tree ):
    corpus = tree.getroot()
    raw_output = corpus.findall("sentence")
    train_output = []
    for sentence_section in raw_output:
        text_complete = sentence_section.find('text').text.lower()
        text_words = text_to_word_sequence( sentence_section.find('text').text, lower=True)
        aspect_terms_root = sentence_section.find('aspectTerms')
        indices = np.zeros( MAX_SEQ_LENGTH )
        # some sentences do not have aspect terms
        if aspect_terms_root:
            aspect_terms_sub = aspect_terms_root.findall('aspectTerm')
            if len(aspect_terms_sub) > 0:
                for aspect_term in aspect_terms_sub:
                    try:
                        term_words = text_to_word_sequence( aspect_term.attrib['term'])
                        for term_word in term_words:
                            term_index =  text_words.index(term_word.lower())
                            indices[term_index] = 1

                    except:
                        print()
                train_output.append( indices )
            else:
                train_output.append( indices )
        else:
            train_output.append( indices )


    return train_output

def get_paper_model( input_shape ):
    model = Sequential()
    # first layer with 100 feature map with filter size 2
    model.add( layers.Conv1D( 100, 2,strides=1, input_shape=input_shape))
    model.add( Activation( 'tanh' ) )
    model.add( MaxPooling1D( pool_size=2, strides=1 ) )
    # first layer with 100 feature map with filter size 2
    model.add( layers.Conv1D( 50, 3, strides=1,kernel_regularizer=l2(0.01) ) )
    model.add( Activation( 'tanh' ) )
    model.add( MaxPooling1D( pool_size=2, strides=1  ) )
    model.add( layers.Dropout( 0.2 ) )
    model.add( layers.Flatten())
    model.add( layers.Dense( MAX_SEQ_LENGTH, activation='softmax') )
    model.add( Activation( 'softmax' ) )
    model.compile( loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'] )
    return model


if __name__ == "__main__":
    MAX_SEQ_LENGTH = 65
    EMBEDDING_SIZE = 300
    tree = ET.parse(  "laptops-trial.xml" )
    sentences = get_sentences( tree )
    sentences_processed = process_sentences( sentences )
    train_out = get_output( tree )

    model = get_paper_model( (MAX_SEQ_LENGTH,EMBEDDING_SIZE))

    print( sentences[1])
    print( train_out[1])
