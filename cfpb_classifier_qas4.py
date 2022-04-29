"""QAS4 CASE STUDY MODEL DEV SCRIPT 

- this script is used to generate candiate and champion models and evalute model performance 
- all outputs are saved and visualized via jupyer 
- final model is saved as a .job file and can be loaded and scored on new data that passes through helpers.pre_process.preprocess_data()
"""
####QAS 4 CASE STUDY 
####

import os, pickle
from tabnanny import verbose
import pandas as pd

#custom 
from data.get_data import source_data
from helpers.eda import run_eda
from helpers.pre_process import generate_StopWords_Phrases, preprocess_data
from helpers.pipeline_ops import build_bow_models

from sklearn import model_selection
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier,AdaBoostClassifier
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline,make_pipeline,FeatureUnion

#text preprocesing requirments
excluded_punct={'+', ':', '[', '^', '"', '|', '{', '@', '=', ')', '%', '#', '+','`', '}', "'", '(', ','
                , '!', '*', '_', '>', '?', '&', '-', '~', '\\', '<', '/', '.', ']', ';', '$'}

stopwords = stopwords.words('english')
newStopWords = ['.','?','%','Google','Wells Fargo','guggenheim partners llc','new york','guggenheim partners'
                ,'bank america','wells fargos','year','thing','would','include','tuesday','make','time','state','bank'
                ,'certain','country','string','perhaps','Donald Trump','Charles Schwab','Morgan Stanley','Credit Suisse'
                , 'Reuters','Bank of America','Guggenheim','Deutsch Bank','Goldman Sachs','Facebook','Fifth Third Bank'
                ,'New York','Washington','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday','January','February'
                ,'March','April','May','June','July','August','September','October','November','December','from', 'subject']

#stop words and phrases to remove
stopwords.extend(newStopWords)
stop_list=set(stopwords)
stop_phrases=generate_StopWords_Phrases(stop_list)[1]
stop_terms= generate_StopWords_Phrases(stop_list)[0]


####################
#source data inputs#
###USER INPUTS######
data_url = 'https://files.consumerfinance.gov/ccdb/complaints.csv.zip'
project_folder = 'C:\\Users\\zjc10\\Desktop\\Projects\\code\\qas4_casestudy'
text_colname = 'text'
target = 'product_group'
eda = False
hp = True               #if True, assumed BOW models already build and chamption model has been selected to go through HP tuning
keras_seq_flag = False  # flag to indicate if the model takes a sequence of ordered terms
                        # if TRUE, do NOT remove censored words(order of censored words in sequence matters)
                        # if FALSE, model is baseline bow based input, remove the XXXXX censored words, order is lost)

#############
#SOURCE DATA#
#############

#clean the data (this is the same process to apply to test data, so we functionalize it to make it re-usable
if not os.path.isfile(project_folder+'cfpb_postbowfilter.pickle'):
    df = source_data(data_url, project_folder)
    df = preprocess_data(df,text_colname, stop_phrases,  stop_list ,project_folder = project_folder)
else:
    with open(project_folder+'cfpb_postbowfilter.pickle', "rb") as input_file:
        df = pickle.load(input_file)


#######
##EDA##
#######
if eda is True:
    run_eda(df,text_colname,target)

###########
##BASELINE SKLEARN PIPELINE MODEL SELECTION 
##########
if not hp: 
    pipe_fit ,feats,  model_metrics, data_dic , cm_stats, preds = build_bow_models(df
                                                                                , target='product_group'
                                                                                , project_folder = project_folder)

    
    #NOTE: ALL ANALYSIS OF MODEL OUTPUT AND PERFORMANCE METRICS CONDUCTED IN JUPYTER FOR EASE OF USE 

if hp:
    ########################################
    ####HYPER PARAM TUNING FOR BEST MODEL###
    #################GBM####################
    from sklearn.model_selection import cross_val_score
    from helpers.pipeline_ops import get_feats, get_data_dic
    import numpy as np
    
    #basic gbm setup which will be enriched through hp tuning
    data_dic = get_data_dic(df)
    feats = get_feats()
    pipe_gbm = Pipeline([('features', feats),('clf', GradientBoostingClassifier())])

    # logistic hyper parameters
    hyperparameters = {
        'clf__n_estimators': [50, 100, 200,300] ,
        'clf__min_samples_split': [2, 5, 10] ,
        'clf__min_samples_leaf': [1, 2, 5] ,
        'clf__max_features': ['auto',200, 300,400] ,
        'clf__max_depth': [5, 10, 20] ,
        'clf__subsample': [1,.7,.5]
    }

    # execute hyperparameter turning , leveraging oob_score to evaluate performance during training
    # AND execute cross validation with a random holdout from training
    # NOTE: BECAUSE OF RESOURCE AND TIME CONTRAINTS I AM USING RANDOMGRIDSEARCH to evaluate the
    # largest breadth of options with a set number of runs
    hp_clf = model_selection.RandomizedSearchCV(pipe_gbm
                       , hyperparameters
                       , cv=3
                       , scoring='balanced_accuracy'
                       , refit=True
                       , n_iter=10
                       , n_jobs=4
                       , verbose=100)  # ,n_jobs=jobs)

    # Fit and id optimal params
    hp_clf.fit(data_dic['training']['x'], data_dic['training']['y'])
    optimized_parms = {k.replace('clf__',''): v for k,v in hp_clf.best_params_.items()}
    optimized_clf = hp_clf.best_estimator_
    cv_metrics = pd.DataFrame(hp_clf.cv_results_)

    preds = hp_clf.predict(data_dic['validation']['x'])
    probs = hp_clf.predict_proba(data_dic['validation']['x'])
    cf = metrics.confusion_matrix(data_dic['validation']['y'], preds)
    print('mean accuracy:{}'.format(np.mean(preds == data_dic['validation']['y'])))

    #save cv metrics 
    with open('C:\\Users\\zjc10\\Desktop\\Projects\\code\\qas4_casestudy\\data\\cv_metrics.pickle', 'wb') as handle:
        pickle.dump(cv_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
   
    with open('C:\\Users\\zjc10\\Desktop\\Projects\\code\\qas4_casestudy\\data\\hp_clf.pickle', 'wb') as handle:
        pickle.dump(hp_clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('C:\\Users\\zjc10\\Desktop\\Projects\\code\\qas4_casestudy\\data\\data_dic_gbm.pickle', 'wb') as handle:
        pickle.dump(data_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('C:\\Users\\zjc10\\Desktop\\Projects\\code\\qas4_casestudy\\output\\gbm_val_preds.pickle', 'wb') as handle:
        pickle.dump({'preds':preds,'probs':probs}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('C:\\Users\\zjc10\\Desktop\\Projects\\code\\qas4_casestudy\\output\\gbm_cm.pickle', 'wb') as handle:
        pickle.dump( cf, handle, protocol=pickle.HIGHEST_PROTOCOL)
                      

    # save model to file
    import joblib
    import numpy as np
    filename = 'C:\\Users\\zjc10\\Desktop\\Projects\\code\\qas4_casestudy\\models\\gridsearch_gbm_obj.sav'
    joblib.dump(hp_clf, filename)

    # fit the optimized model (try using the clf output object, if fails, upback the optimal params found from gridsearch)
    final_gbm = optimized_clf.fit(data_dic['validation']['x'], data_dic['validation']['y'])

    # refitting ALL DATA on final model specification
    # view predictions and probabilities of each class
    x = pd.concat([data_dic['training']['x'] , data_dic['validation']['x']]) 
    y = pd.concat([data_dic['training']['y'],data_dic['validation']['y']])
    final_gbm.fit(x, y)

    preds = final_gbm.predict(x)
    probs = final_gbm.predict_proba(x)
    cf = metrics.confusion_matrix(y, preds)
    print('mean accuracy:{}'.format(np.mean(preds == y)))
  
    #save it 
    with open('C:\\Users\\zjc10\\Desktop\\Projects\\code\\qas4_casestudy\\output\\final_gbm_preds.pickle', 'wb') as handle:
        pickle.dump(preds, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('C:\\Users\\zjc10\\Desktop\\Projects\\code\\qas4_casestudy\\output\\final_gbm_probs.pickle', 'wb') as handle:
        pickle.dump(probs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save model to file
    import joblib
    filename = 'C:\\Users\\zjc10\\Desktop\\Projects\\code\\qas4_casestudy\\models\\final_baseline_model_gbm.sav'
    joblib.dump(final_gbm, filename)

