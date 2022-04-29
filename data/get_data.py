#!/usr/bin/env python

import pandas as pd
import os ,sys,urllib
from pathlib import Path
from zipfile import ZipFile

#source data 
def source_data(data_url, project_folder):

    #download the data
    zip_path = os.path.join(project_folder,'data','complaints.csv.zip')

    if os.path.isfile(os.path.join(project_folder,'data','complaints.csv.zip')) is False:
        urllib.request.urlretrieve(data_url,zip_path)
        with ZipFile(zip_path,'r') as zipp:
            zipp.extractall(path = os.path.join(project_folder,'data'))
            
    return get_case_study_data(zip_path.replace('.zip',''))
    

def get_case_study_data(data_path):
    """
    Function to match complaint narratives to message ids used in case study
    """
    results_df = pd.read_csv(data_path)
    results_df.columns = [c.lower().replace(' ', '_') for c in results_df.columns]
    results_df = results_df[['complaint_id', 'consumer_complaint_narrative']]
    results_df.rename(columns={'consumer_complaint_narrative':'text'}, inplace=True)
    results_df.complaint_id = results_df.complaint_id.astype('int64')

    # Load Case Study message ids
    path_p = Path(data_path).parent.absolute()
    case_study_df = pd.read_csv(os.path.join(path_p,"case_study_msg_ids.csv"))
    case_study_df.complaint_id = case_study_df.complaint_id.astype('int64')

    # Join by msg_id
    case_study_df = case_study_df.merge(results_df, on='complaint_id', how='left')
	
	# Drop NAs
    case_study_df.dropna(inplace=True)

    return case_study_df
