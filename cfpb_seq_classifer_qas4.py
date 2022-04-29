import os,sys, urllib
import pandas as pd
import matplotlib.pyplot as plt
from os import path
import pymagnitude

#custom
from data.get_data import source_data
from helpers.eda import plot_target_dist
from helpers.pre_process import generate_StopWords_Phrases, encode_target_alt, preprocess_data
from helpers.pipeline_ops import TextSelector, sk_model_stats
from helpers.seq2seq_helpers import make_df,make_embeddings_pymag, precision, recall,fbeta_score,fmeasure, build_model
from nltk.corpus import stopwords
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

#text preprocesing
excluded_punct={'+', ':', '[', '^', '"', '|', '{', '@', '=', ')', '%', '#', '+','`', '}', "'", '(', ',', '!', '*', '_', '>', '?', '&', '-', '~', '\\', '<', '/', '.', ']', ';', '$'}
stopwords = stopwords.words('english')
newStopWords = ['.','?','%','Google','Wells Fargo','guggenheim partners llc','new york','guggenheim partners'
                ,'bank america','wells fargos','year','thing','would','include','tuesday','make','time','state','bank'
                ,'certain','country','string','perhaps','Donald Trump','Charles Schwab','Morgan Stanley','Credit Suisse'
                , 'Reuters','Bank of America','Guggenheim','Deutsch Bank','Goldman Sachs','Facebook','Fifth Third Bank'
                ,'New York','Washington','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday','January','February'
                ,'March','April','May','June','July','August','September','October','November','December','from', 'subject'
                , 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice'
                , 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right'
                , 'line', 'even', 'also', 'may', 'take', 'come','the']

#stop words and phrases to remove
stopwords.extend(newStopWords)
stop_list=set(stopwords)
stop_phrases=generate_StopWords_Phrases(stop_list)[1]
stop_terms= generate_StopWords_Phrases(stop_list)[0]


#source data inputs
data_url = 'https://files.consumerfinance.gov/ccdb/complaints.csv.zip'
project_folder = 'C:\\Users\\zjc10\\Desktop\\Projects\\code\\qas4_casestudy'
text_colname = 'text'
target = 'product_group'

#static paths
DATAPATH="C:\\Users\\zjc10\\Desktop\\Projects\\code\\qas4_casestudy\\data\\"
OUTPUTPATH =os.path.join(project_folder,'output')
PYMAGPATH = "C:\\Users\\zjc10\\Desktop\\Utils\\embeddings\\crawl-300d-2M.magnitude"
filename = os.path.join(project_folder,'data',"cfpb_complaints.csv")
MODELPATH = "C:\\Users\\zjc10\\Desktop\\Projects\\code\\qas4_casestudy\\Models\\"

#FASTTEXT PYMAG EMBEDDINGS (optimized file format to query word embedding vector space, enables OK performance with cpu)
wv=pymagnitude.Magnitude(PYMAGPATH)
max_features = 100000 #maximum number of words to consider in analysis
EMBEDDING_DIM = 300 #NOTE: becuase we use pymag this must be set to 300

#flag to indicate if the model takes a sequence of ordered terms
# if TRUE, do NOT remove censored words(order of censored words in sequence matters)
# if FALSE, model is baseline bow based input, remove the XXXXX censored words, order is lost)
keras_seq_flag = True

#############
#SOURCE DATA#
#############
#clean the data (this is the same process to apply to test data, so we functionalize it to make it re-usable
if not os.path.isfile(project_folder+'cfpb_postbowfilter.pickle'):

    #pull data an unzip
    df = source_data(data_url, project_folder)

    # clean the data (this is the same process to apply to test data, so we functionalize it to make it re-usable
    df = preprocess_data(df
                        , text_colname
                        , stop_phrases
                        , stop_list
                        , keras_seq_flag= keras_seq_flag 
                        , project_folder = project_folder)

    #encode target
    df, targDefs = encode_target_alt(df, target='Product')

else:
    import pickle
    with open(project_folder+'cfpb_postbowfilter.pickle', "rb") as input_file:
        df = pickle.load(input_file)
        df = df.groupby('labelid', group_keys=False).apply(lambda x: x.sample(4000))



#create sequences
xtr,ytr,xte,yte,word_index, maxlen,words = make_df(df
                                                   , max_features
                                                   , EMBEDDING_DIM
                                                   , stop_list
                                                   , 'labelid'
                                                   , rebalance_data=False
                                                   , test_size=.2
                                                   , textcol='finalcomplaintstring')


#######################################################
##############BEGIN MODEL DEVELOPMENT##################
#######################################################
#GENERATE EMBEDDING MATRIX FOR EMBEDDING LAYER IN MODEL(using pymagnitude converted vectors)
nb_words = min(max_features, len(word_index)+1) #total features to consider, min of the max feats vs total feats
embedding_vector =  make_embeddings_pymag(wv, max_features,words, word_index, EMBEDDING_DIM, nb_words)



#BEGIN MODEL LAYER DEFFINITIONS
#Initialize Sequential Model class
model = build_model( EMBEDDING_DIM,embedding_vector,maxlen,nb_words, targ_levels = 7, transfer_learn = True )


#SPECIFY KERAS CALLBACKS
filepath = MODELPATH+"best_bilstm.hdf5"
monitor = 'val_acc'#'val_fmeasure'
min_max_metric = 'max'
ckpt = ModelCheckpoint(filepath, monitor=monitor , verbose=1, save_best_only=True, save_weights_only=False, mode=min_max_metric)
early = EarlyStopping(monitor=monitor , mode=min_max_metric, patience=2)



#FIT MODEL
model_out = model.fit(xtr, ytr, batch_size=80, epochs=10
                      , validation_data = (xte,yte)
                      #, class_weight=ClassWeights
                      , callbacks=[ckpt, early])


#EVALUATE MODEL FIT
from sklearn import metrics
y_preds = model.predict_classes(xte)
matrix = metrics.confusion_matrix(yte, y_preds)
scores = model.evaluate(xte, yte, verbose=1)
print(matrix)
print(scores)

#save the results
import pickle
with open(f'{project_folder}\\data\\xtr_lstm.pickle', 'wb') as handle:
    pickle.dump(xtr, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'{project_folder}\\data\\ytr_lstm.pickle', 'wb') as handle:
    pickle.dump(ytr, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'{project_folder}\\data\\xtest_lstm.pickle', 'wb') as handle:
    pickle.dump(xte, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'{project_folder}\\data\\ytest_lstm.pickle', 'wb') as handle:
    pickle.dump(yte, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open(f'{project_folder}\\output\\cm_lstm_matrix.pickle', 'wb') as handle:
    pickle.dump(matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'{project_folder}\\output\\lstm_scores.pickle', 'wb') as handle:
    pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'{project_folder}\\output\\lstm_preds.pickle', 'wb') as handle:
    pickle.dump(y_preds, handle, protocol=pickle.HIGHEST_PROTOCOL)