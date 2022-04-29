# qas4_casestudy
cfpb code base used to evaluate BOW and embedding based model architectures
model file and code for cfpb case study. Model file is a h5 file of model output weights that must be loaded using load_model() function from helpers.py

- **CFPB USE CASE VISUAL.ipynb**: Jupyter notebook containing all EDA and visuals of model performance metrics, hyper parameter and cv results, champion BOW model analysis , and sequence model analysis. This notebook also contains the code required to load both the final BOW based GBM model and the sequence based embedding model. 
-
- **helpers.eda.py**: basic functions used to generate eda overview of cfpb dataset 
- **helpers.pipeline_opds.py:** Functions supporting the development of BOW scikit learn model pipelines and evaluation of hyperparam tuning'
-       NOTE: hp_flag must be set to TRUE to execute HP tuning  
- **helpers.seq2seq_helpers.py**: helper functions for building keras sequential api lstm embedding model
- **cfpb_classifier_qas4.py**: All code required to build 1) pipeline of inital candiate BOW models(Logistic, GBM , SVM), 2)Hyperparameter tuning and CV using champion GBM BOW model, 3) code required to generate train and save the FINAL binary model file containing the tuned GBM model.
- **cfpb_seq_classifer_qas4.py**:  code required to generate seq2seq model with bi-lstm embedding architecture in keras. this code generates the binary file in models/best_bilstm.hdf5, and can be loaded directly into the jupyter notebook for scoring new records.
- **models/best_bilstm.hdf5**: binary file for the champion sequence model architecture. there is an example of how to load this binary file in the juupyter notebook containing all the presentation material (CFPB USE CASE VISUALS.ipynb)
- **models/final_baseline_model_gbm.sav**: binary file for the chamption bow based GBM model. there is an example of how to load this binary file in the jupyer notebook containing all the presentation material.  

- **challenger sklearn model pipelines:** baseline sklearn classifers with full e2e preprocess and tuning.ipynb
    - Note: Not all challenger models were able to  run within the time of the use case. however, the full e2e code base holistically illistrates the methodology used for preprocessing , scoring, and hyperparameter tuning, as well as easy model deployment via pipelines
    
