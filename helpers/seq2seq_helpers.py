import tensorflow.keras as keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import backend as K

import pandas as pd
from nltk.tokenize import word_tokenize , sent_tokenize
import gensim
from sklearn import preprocessing , model_selection,metrics
from imblearn.over_sampling import SMOTE
import numpy as np

# LOAD BEST MODEL AND COMPARE FIT TO LAST EPOCH
def load_best_model(model_path=None, xtr=None, xte=None, ytr=None, yte=None,
                    custom_obs={'SeqSelfAttention':None}):
    model_path = model_path
    model_out = keras.models.load_model(model_path, custom_objects=custom_obs)

    # GET CONFUSION MATRIX FOR BEST MODEL
    y_preds = model_out.predict_classes(xtr)
    y_preds_test = model_out.predict_classes(xte)

    matrix = metrics.confusion_matrix(ytr, y_preds)
    matrix_test = metrics.confusion_matrix(yte, y_preds_test)

    print('Final Model Confusion Matrix')
    print(matrix)
    print('Final Model Confusion Matrix')
    print(matrix_test)

    return model_out, matrix, matrix_test



#####MODEL DATA PREPROCESSING FUNCTION
def make_df(datapath
            , max_features
            , EMBEDDING_DIM
            , stop_list
            , target
            , rebalance_data=False
            , test_size=.2
            , textcol='final_complaint_string'):
    '''
    Inputs:
    ##### datapath = path to dataframe of sentences and label ids
    ##### max_features = total number of features from text to consider when buildling embedding layer
    ##### EMBEDDING_DIM = total number of dimensions from embedding we want to use in embedding layer (Note: if using pymag you must use 300)
    ##### stop_lsit = list of stopwords to use for preprocessing
    ##### stemmer = stemming class to use to stem words in sentences
    ##### target = name of the target feild you are trying to predict in the dataset
    PROCESS STEPS:
    ##### 1. toeknize text
    ##### 2. Convert text to sequences
    ##### 3. Pad Sequences
    ##### 4. Split into Test and Train
    ##### 5. Return X_train , X_Test, Y, WordIndex
    '''

    # load data
    if isinstance(datapath, object):
        data = datapath
    else:
        data = pd.read_pickle(datapath)

    # clean original sentence level text
    # trainDF = orig_text_clean(data,target=target , txtfeild='Sentence',stopwords=stop_list,stemmer=stemmer)

    # specify X and Y (these standard columns are created in orig_text_clean)
    X = data[textcol]
    Y = data[target]

    # generate list of unique words based on total words specified to consider
    sentences = []
    lines = X.values.tolist()
    lines = [word_tokenize(sent) for sent in lines]
    model = gensim.models.Word2Vec(sentences=lines, size=EMBEDDING_DIM, window=5, workers=4, min_count=1)
    words = list(model.wv.vocab)

    # model process
    # fit tokenizer to words and turn to sequences
    tokenizer_obj = Tokenizer()
    tokenizer_obj.fit_on_texts(X)
    sequences = tokenizer_obj.texts_to_sequences(X)

    # define max length for padding and total vocab size
    max_length = max([len(s.split()) for s in X])
    vocab_size = len(tokenizer_obj.word_index) + 1

    # pad sequences
    word_index = tokenizer_obj.word_index
    review_pad = pad_sequences(sequences, maxlen=max_length)
    label = Y.values

    # split data
    x_train, x_test, y_train, y_test = model_selection.train_test_split(review_pad, label, shuffle=True, stratify=Y,
                                                                        test_size=test_size, random_state=10)

    if rebalance_data == True:
        sm = SMOTE(random_state=2)
        x_train, y_train = sm.fit_sample(x_train, y_train)

    return x_train, y_train, x_test, y_test, word_index, max_length, words



#pymagnitude vectors

def make_embeddings_pymag(wv, max_features,words, word_index, embed_size, nb_words):
    #embeddings using pymagnitude
    embeddings_index = {}
    for w in words:
        word =w
        coefs = wv.query(word) #np.asarray(values[1:])
        embeddings_index[word]=coefs

    embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


#peformance metrics
def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    # Calculates the F score, the weighted harmonic mean of precision and recall.

    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    return fbeta_score(y_true, y_pred, beta=1)




def build_model( embedding_dim, embedding_vector, maxlen,nb_words,targ_levels=7, transfer_learn=True):
    model = keras.models.Sequential()

    # add embedding layer
    model.add(keras.layers.Embedding(input_dim=nb_words  # max_features
                                     , output_dim=embedding_dim  # embedding size
                                     , weights=[embedding_vector]
                                     , mask_zero=False
                                     , trainable=transfer_learn
                                     # if True transfer learning is enabled, the weights from the past epoch are used as starting points for the next epoch
                                     , input_length=maxlen
                                     # the max sequence length, this is the length that all sequences will be padded to (so all sequences will have same length)
                                     ))

    # ADD CNN LAYER WITH SELU ACTIVATION AND GAUSSIAN DROPOUT
    # NOTE:specify guassian dropout befire activation function has been specified
    # WARNING: if you specify guassian dropout after activation function  the resulting values may be out-of-range
    # from what the activation function may normally provide. For example, a value with added noise may be less than zero,
    # whereas the relu activation function will only ever output values 0 or larger.
    model.add(keras.layers.Conv1D(filters=164, kernel_size=10, padding='valid', strides=1))
    model.add(keras.layers.MaxPooling1D(pool_size=4))
    model.add(keras.layers.GaussianDropout(0.2))
    model.add(keras.layers.Activation('selu'))

    # ADD BILSTM WITH DROPOUT
    # DROPOUT VERY HIGH TO AVOID OVERFIT
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=100,
                                                           return_sequences=True,
                                                           # dropout=.3, #dropout not applied here becase we apply it via the guassian dropout specificed below
                                                           recurrent_dropout=.2,
                                                           recurrent_regularizer=keras.regularizers.L1L2(l1=0.0,
                                                                                                         l2=0.01),
                                                           kernel_regularizer=keras.regularizers.L1L2(l1=0, l2=.01),
                                                           bias_regularizer=keras.regularizers.L1L2(l1=0, l2=.01),
                                                           )))
    # ADD NON RECURRENT DROPUT TO LSTM
    model.add(keras.layers.GaussianDropout(0.2))
    model.add(keras.layers.Activation('selu'))

    # FLATTENED LAYER (CONVERTS 2D tensor to 1D)
    # model.add(GlobalMaxPool1D())
    model.add(keras.layers.Flatten())

    # OUTPUT LAYER(units = # target labels, for binary this =1)
    model.add(keras.layers.Dense(units=targ_levels, activation='softmax'))

    # COMPILE MODEL WITH CUSTOM KERAS METRICS (functions defined at top of script)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])#, recall, fmeasure, precision, fbeta_score


    return model