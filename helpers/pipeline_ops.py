from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import metrics
import scikitplot as skplt


import pandas as pd

#custom
import sklearn
from .pre_process import  encode_target_alt
from imblearn.over_sampling import RandomOverSampler
from sklearn import svm
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier,AdaBoostClassifier
from sklearn.pipeline import Pipeline,make_pipeline,FeatureUnion
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

class TextSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on text columns in the data
    """

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]

    def get_feature_names(self):
        return self.key


from statistics import mean
import numpy as np


def plotmetrics(valid_y, pred_probs, preds):
    return [
        skplt.metrics.plot_roc(valid_y, pred_probs)  # ;plt.show()
        #, skplt.metrics.plot_ks_statistic(valid_y, pred_probs)  # ;plt.show()
        , skplt.metrics.plot_precision_recall(valid_y, pred_probs)  # ;plt.show()
        #, skplt.metrics.plot_cumulative_gain(valid_y, pred_probs)  # ;plt.show()
        #, skplt.metrics.plot_lift_curve(valid_y, pred_probs)  # ;plt.show()
        #, skplt.metrics.plot_confusion_matrix(valid_y, preds, normalize=True)  # ;plt.show()
        # feature_importances=pipe_rf.named_steps['clf'].feature_importances_
        # skplt.estimators.plot_learning_curve(pipe_rf, train_x, train_y);plt.show()
    ]


def sk_model_stats(pipe_dict, x, y, data_type='train', plot=False):
    """ get model performance metrics from each sklearn pipe"""
    cm_stats = {}
    preds_out = {}
    summarys = []
    x = x
    y = y
    for key, value in pipe_dict.items():
        preds = value.predict(x)
        prob_preds = value.predict_proba(x)
        print('Model Validation Charts {}:'.format(key))
        if plot is True:
            plotmetrics(y, prob_preds, preds)

        cm = metrics.confusion_matrix(y, preds).flatten()
        df_dict = {'model': key
            , 'data': data_type
            , 'logloss': metrics.log_loss(y, prob_preds)
            , 'accuracy_score': metrics.accuracy_score(y, preds)
            , 'recall': metrics.recall_score(y, preds,average = 'weighted')
            , 'precision': metrics.precision_score(y, preds,average='weighted')
            , 'f1_score': metrics.f1_score(y, preds, average='weighted')
            , 'fbeta_score': metrics.fbeta_score(y, preds, beta=.75,average='weighted')
            , 'mae': metrics.mean_absolute_error(y, preds)
            , 'mse': metrics.mean_squared_error(y, preds)
                                }

        summarys.append(pd.DataFrame(df_dict, index=[0]))
        cm_stats[key] = cm
        preds_out[key] = {'preds':preds , 'prob_preds':prob_preds}

    appended_data = pd.concat(summarys,axis=0)
    out = appended_data.sort_values(by=['f1_score'], ascending=False)
    return out  , cm_stats, preds_out


#get pipeline object to process 
def get_feats():
    #define text transformations to generate
    #apply text_feats transformations iterativly into tfidf tranformation pipeline
    #note: x is the name of the feature vector output from train_test_split, the column actually being used is finalcomplaintstring (becuase tfidf requires string input)
    from sklearn.pipeline import Pipeline
    text_feats = {'tfidf_pipe':{'analyzer':'word','ngram_range':(1,1),'max_features':1000,'min_df':.01,'max_df':.08}}
    text_pipe = {key : Pipeline([('selector', TextSelector(key='x')),
                                ('tfidf',  TfidfVectorizer(analyzer=val['analyzer']
                                                        , ngram_range=val['ngram_range']
                                                        , max_features=val['max_features']
                                                        , min_df=val['min_df']
                                                        , max_df=val['max_df']
                                                        , smooth_idf=True
                                                        , norm='l2'
                                                        , sublinear_tf=True
                                                        , lowercase=True))

                                ]) for key,val in text_feats.items()
                        }

    #THIS IS SET UP TO HANDLE FUTURE ENHANCEMENTS TO FEATURE SPACE, WE CAN CONCATENATE NUMERICAL and OTHER CATEGORICAL FEATURES USING FEATURE UNION
    #NORMALLY I WOULD INCLUDE CATAGORTICAL TEXT FEATURES AND NUMERIC FEATURES, WHICH REQUIRES A FEATURE UNION
    #HERE I AM JUST USING FEATURE UNION TO FORMAT THE PIPELINE CORRECTLY
    #make a pipeline from all of our pipelines, we do the same thing, but now we use a FeatureUnion to join the feature processing pipelines.
    final_pipeline = [(k,v) for k, v in text_pipe.items()]
    feats = FeatureUnion(final_pipeline)
    return feats

def get_data_dic(df:pd.DataFrame):
    #split data and create dictonary
    df = df.rename(columns={'finalcomplaintstring':'x'})
    train_x,valid_x,train_y,valid_y= model_selection.train_test_split(df[['complaintid','x']] #keep complaint id for tracking
                                                                    , df['labelid']
                                                                    , shuffle=True
                                                                    , stratify=df['labelid']
                                                                    , test_size=.2
                                                                    , random_state=10
                                                                    )
    data_dic={'training':{'x':train_x,'y':train_y},'validation':{'x':valid_x,'y':valid_y}}
    return data_dic

#generate models for use in comparison to seq model
def build_bow_models(df: pd.DataFrame
                     , target: str ='product_group'
                     ,project_folder: str = None
                     ,n_jobs: int = 8
                     ):

    assert 'finalcomplaintstring' in list(df), 'finalcomplaintstring is the transformed text column required for scoring, please rerun preprocessing to derive the field'

    #encode target
    if 'labelid' not in list(df):
        df ,targDefs= encode_target_alt(df,target=target)

    ##################
    #SET UP PIPELINES#
    ##################
    feats = get_feats()

    #NOTE: OVERSAMPLING TO ACCCOUNT FOR CLASS IMBALANCE, NOT THE MOST EFFECTIVE APPROACH BUT FAST AND ACKNOWLODGES CLASS IMBALANCE NEEDS TO BE ADDERSSSED
    #ALTERNATE APPROACHES: COLLAPSE LEVELS OF TARGET, GENERATE N-BINARY MODELS, 1 for each level of target
    from imblearn.pipeline import Pipeline
    model_dic = {'pipe_log': [('features', feats)
                              , ('oversamp', RandomOverSampler())
                              , ('feature_selection', SelectFromModel(ExtraTreesClassifier(n_estimators=500, n_jobs=n_jobs,max_features='auto')))
                              , ('clf', LogisticRegression(random_state=42, max_iter=200, verbose=True,n_jobs = n_jobs))]

                , 'pipe_rf': [('features', feats)
                              , ('oversamp', RandomOverSampler())
                              , ('clf', RandomForestClassifier(n_estimators=200
                                                             , max_features='auto'
                                                             , oob_score=True #CV
                                                             , max_depth=10
                                                             , bootstrap = True 
                                                             , n_jobs=n_jobs, verbose=True))]

                , 'pipe_gbm': [('features', feats)
                                , ('oversamp', RandomOverSampler())
                                , ('feature_selection', SelectFromModel(ExtraTreesClassifier(n_estimators=200, n_jobs=n_jobs,max_features='auto')))
                                , ('clf', GradientBoostingClassifier(verbose=True,n_estimators = 200 ,min_samples_leaf=5,validation_fraction=.1)) #oob cv
                                                ]
                 }

    #pipeline dic(key=Model name , value = model pipeline)
    #USING IMBD PIPELINE TO ENABLE OVERSASMPLING
    pipe_dic = {key:Pipeline([process for process in value]) for key,value in model_dic.items()}

    data_dic = get_data_dic(df)

    #FIT EACH MODEL
    pipe_fit = {key:value.fit(data_dic['training']['x']
                              , data_dic['training']['y'])
                for key,value in pipe_dic.items()}


    #GENERATE PERFORMANCE METRICS ON TRAIN AND VALIDATION DATASETS
    train_metrics , train_cm_stats , train_preds = sk_model_stats(pipe_fit,data_dic['training']['x']
                                                                  , data_dic['training']['y'],data_type='train')

    valid_metrics , valid_cm_stats , valid_preds = sk_model_stats(pipe_fit,data_dic['validation']['x']
                                                                  , data_dic['validation']['y'],data_type='valid')

    model_metrics=train_metrics.append(valid_metrics)
    cm_stats = {'train':train_cm_stats,'valid':valid_cm_stats}
    preds = {'train': train_preds, 'valid':valid_preds}

    print(model_metrics)
    #save the results
    import pickle
    with open(f'{project_folder}\\models\\model_fits_.pickle', 'wb') as handle:
        pickle.dump(pipe_fit, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{project_folder}\\data\\model_ready_data_.pickle', 'wb') as handle:
        pickle.dump(data_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{project_folder}\\output\\model_metrics_.pickle', 'wb') as handle:
        pickle.dump(model_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{project_folder}\\output\\cm_stats_.pickle', 'wb') as handle:
        pickle.dump(cm_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{project_folder}\\output\\preds_.pickle', 'wb') as handle:
        pickle.dump(preds, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{project_folder}\\output\\feats_.pickle', 'wb') as handle:
        pickle.dump(feats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return pipe_fit, feats, model_metrics , data_dic, cm_stats, preds





def load_and_score(model_path,data2score):
    import joblib
    
    #Final Prediction on entire dataset
    #load model 
    loaded_model = joblib.load(model_path)
    
    #len(newdata.loc[newdata.index.duplicated(keep='first')])
    newdata = orig_text_clean(data)
    
    #preds on new data 
    predictions = loaded_model.predict_proba(newdata)
    preds = pd.DataFrame(data=predictions, columns = loaded_model.classes_)
    
    #generating a submission file
    results = pd.concat([newdata, preds], axis=1)
    return results