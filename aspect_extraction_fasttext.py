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
from sklearn.metrics import accuracy_score
from gensim.models import KeyedVectors
import collections



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
    model.add( layers.Conv1D( 200, 2,strides=1, input_shape=input_shape))
    model.add( Activation( 'tanh' ) )
    model.add( MaxPooling1D( pool_size=2, strides=1 ) )
    model.add( layers.Conv1D( 100, 3, strides=1,kernel_regularizer=l2(0.01) ) )
    model.add( Activation( 'tanh' ) )
    model.add( MaxPooling1D( pool_size=2, strides=1  ) )
    model.add( layers.Dropout( 0.2 ) )
    model.add( layers.Flatten())
    model.add( layers.Dense( MAX_SEQ_LENGTH, activation='softmax') )
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)
    model.compile( loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'] )
    model.summary()
    return model


def compute_accuracy_once(y,y_pred):
    accuracy = 0
    correct_ones=0
    if 1 in y[:]:
        counter = collections.Counter(y)
        for i,value in enumerate(y):
            if y[i]==y_pred[i] and y[i]==1:
                correct_ones+=1
        accuracy = round(correct_ones/dict(counter)[1],2)
    else:
        correct=0
        for i,value in enumerate(y):
            if y[i]==y_pred[i] and y[i]==1:
                correct+=1
        accuracy = round( correct/len(y), 2)
    return accuracy

def compute_accuracy(y, y_pred):
    accuracy_collected = []
    for i, value in enumerate(y):
        accuracy_collected.append( compute_accuracy_once(y[i],y_pred[i]) )
    return np.mean( accuracy_collected )

def get_sentences_as_wordlists( path ):
    sentences  = get_sentences( path )
    sentences_as_word_list = [ text_to_word_sequence(s) for s in sentences]
    return sentences_as_word_list



if __name__ == "__main__":
    MAX_SEQ_LENGTH = 65
    EMBEDDING_SIZE = 300
    TOP_WORDS = 20000
    EPOCHS = 250
    #TRAIN_DATA_PATH = "laptops_train/Laptops_Train.xml"
    TRAIN_DATA_PATH = "datasets/Restaurants_Train.xml"
    FASTTEXT_MODEL = "fasttext_vectors/wiki.en.vec"
    tree = ET.parse( TRAIN_DATA_PATH  )
    sentences = get_sentences( tree )
    sentences_processed = process_sentences( sentences )

    sentences_words = modules.vector_representations.get_sentences_as_wordlists( TRAIN_DATA_PATH )
    en_model = KeyedVectors.load_word2vec_format( FASTTEXT_MODEL )

    X_data = modules.vector_representations.get_fasttext_vectors( sentences_words, MAX_SEQ_LENGTH, EMBEDDING_SIZE, en_model )
    y_data = np.array( get_output( tree ) )


    X_train, X_test, y_train, y_test = train_test_split( X_data, y_data, test_size=0.1, random_state=42)
    model = get_paper_model( (MAX_SEQ_LENGTH,EMBEDDING_SIZE))
    #model.fit( X_train, y_train, epochs=EPOCHS, batch_size=50, validation_data=(X_test, y_test) )
    model.fit( X_data, y_data, epochs=EPOCHS, batch_size=50 )




    y_pred = model.predict(X_test)

    processed_output = []
    for i in range(y_pred.shape[0]):
        processed_label =[]
        for j in range(y_pred.shape[1]):
            if y_pred[i][j] > 0.1:
                processed_label.append(1)
            else:
                processed_label.append(0)
        processed_output.append(processed_label)


    index=50
    print( y_pred[index] )
    print( processed_output[index])
    print( y_data[index])
    accuracy = compute_accuracy( y_test, processed_output )
    print( "ACC: ", accuracy)
