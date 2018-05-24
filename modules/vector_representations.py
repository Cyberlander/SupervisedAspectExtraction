import xml.etree.ElementTree as ET
from keras.preprocessing.text import text_to_word_sequence
from gensim.models import word2vec
from gensim.models import Word2Vec
import gensim.models.keyedvectors as word2vec
from nltk.corpus import stopwords
import numpy as np

def get_sentences( path ):
    tree = ET.parse( path ) #"../laptops-trial.xml" )
    corpus = tree.getroot()
    sentences = []
    sent = corpus.findall(".//sentence")
    for s in sent:
        sentences.append( s.find( 'text' ).text )
    return sentences

def get_sentences_as_wordlists( path ):
    sentences  = get_sentences( path )
    sentences_as_word_list = [ text_to_word_sequence(s) for s in sentences]
    return sentences_as_word_list


def train_word2vec_model( sentences_path ):
    sentences_raw = get_sentences( sentences_path )
    sentences_as_word_list = [ text_to_word_sequence(s) for s in sentences_raw]
    num_features = 300
    min_word_count = 1
    num_workers = 4
    context=5
    downsampling=1e-3
    model = word2vec.Word2Vec( sentences_as_word_list,
        workers=num_workers, size=num_features,
        min_count=min_word_count, window=context,
        sample=downsampling)
    model.init_sims(replace=True)
    model_name = "../300features_1minword_5context"
    model.save( model_name )

def get_word2vec_model( path, is_binary ):
    if is_binary==False:
        model = Word2Vec.load( path )
    else:
        model = word2vec.KeyedVectors.load_word2vec_format(path, binary=True)
    return model

def get_word2vec_vectors( sentences_as_word_lists, seq_len, embedding_dim, word2vec_model ):
    num_sentences = len( sentences_as_word_lists )
    data_matrix = np.zeros((num_sentences,seq_len,embedding_dim))
    for i,sentence in enumerate( sentences_as_word_lists ):
        sentence = sentence[:65]
        for j, word in enumerate( sentence ):
            if word in word2vec_model.wv.vocab:
                word_vector = word2vec_model.wv[word]
            else:
                word_vector = np.ones(300)

            data_matrix[i,j] = word_vector
    return data_matrix

if __name__ == "__main__":
    print()
    train_word2vec_model("../datasets/Restaurants_Train.xml")
    """word2vec_model = get_word2vec_model( path )
    #print(word2vec_model.wv['aluminum'])
    sentences_words = get_sentences_as_wordlists()
    x_data = get_word2vec_vectors( sentences_words, 65, 300, word2vec_model )"""
