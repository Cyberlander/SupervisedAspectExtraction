import xml.etree.ElementTree as ET
from keras.preprocessing.text import text_to_word_sequence
import numpy as np
from keras import layers
from keras.models import Sequential
from keras.layers.pooling import MaxPooling1D
from keras.optimizers import Adam, RMSprop, SGD


MAX_SEQ_LENGTH = 69

def get_sentences( tree ):
    corpus = tree.getroot()
    sentences = []
    sent = corpus.findall(".//sentence")
    for s in sent:
        sentences.append( s.find( 'text' ).text )
    return sentences

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
"""
def get_model( input_shape ):
    model = models.Sequential()
    model.add( layers.Conv1d( 50, kernel_size=(1,5), input_shape=input_shape))
    model.add( MaxPooling1D() )
    model.add(Flatten())
    model.add(Dense(MAX_SEQ_LENGTH))
    model.compile()
    return model"""


if __name__ == "__main__":
    tree = ET.parse(  "laptops-trial.xml" )
    sentences = get_sentences( tree )
    train_out = get_output( tree )

    print( sentences[1])
    print( train_out[1])
