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
from keras.layers.embeddings import Embedding
import modules.vector_representations
from sklearn.model_selection import train_test_split




def get_sentences( tree ):
    corpus = tree.getroot()
    sentences = []
    sent = corpus.findall(".//sentence")
    for s in sent:
        sentences.append( s.find( 'text' ).text )
    return sentences

def process_sentences( sentences ):
    tokenizer = Tokenizer( num_words=TOP_WORDS )
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
    #model.add( layers.Conv1D( 100, 2,strides=1 ))
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
    TOP_WORDS = 20000
    #TRAIN_DATA_PATH = "laptops_train/Laptops_Train.xml"
    TRAIN_DATA_PATH = "datasets/Restaurants_Train.xml"
    WORD2VEC_MODEL = "word2vec_models/restaurant_300features_1minword_5context"
    tree = ET.parse( TRAIN_DATA_PATH  )
    sentences = get_sentences( tree )
    sentences_processed = process_sentences( sentences )

    sentences_words = modules.vector_representations.get_sentences_as_wordlists( TRAIN_DATA_PATH )
    word2vec_model = modules.vector_representations.get_word2vec_model( WORD2VEC_MODEL  )
    X_data = modules.vector_representations.get_word2vec_vectors( sentences_words, MAX_SEQ_LENGTH, EMBEDDING_SIZE, word2vec_model )
    y_data = np.array( get_output( tree ) )


    X_train, X_test, y_train, y_test = train_test_split( X_data, y_data, test_size=0.1, random_state=42)
    model = get_paper_model( (MAX_SEQ_LENGTH,EMBEDDING_SIZE))
    #model.fit( X_train, y_train, epochs=3, batch_size=50, shuffle=True, validation_data=(X_test, y_test) )
    model.fit( X_data, y_data, epochs=30, batch_size=50 )

    index = 10
    print( sentences_words[index])
    print( model.predict(X_data[index].reshape(1,65,300)))
    print( y_data[index])
