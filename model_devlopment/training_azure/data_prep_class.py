
print('print importing lib...')

import argparse
from azureml.core import Run
from azureml.core import Dataset
import joblib
import os
import numpy as np
import pandas as pd
import random
import joblib
from class_recommender import Recommender
from scipy import sparse

print('lib imported...')

#get scripts arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input-data', type=str, dest='dataset_folder')
#outfolder for preprocessed data to use in pipeline. see OutputFileDatasetConfig doc from azure
parser.add_argument('--prepped_data', type=str, dest='prepped_data')
parser.add_argument('--factors', type=int, dest='factors')
parser.add_argument('--regularization', type=float, dest='reg')
parser.add_argument('--iterations', type=int, dest='iterations')
args = parser.parse_args()

#set parameters
save_folder = args.prepped_data
factors = args.factors
reg = args.reg
iterations = args.iterations

#get the experiment run context and workspace
run = Run.get_context()
ws = run.experiment.workspace

#load data
print('loading data...')
#frame = Dataset.get_by_name(ws, dataset_name).to_pandas_dataframe()
# Get the training data path from the input. must be passed as argument in the script as as_named_input + as_download or as_mount
data_path = run.input_datasets['clicks']

#declare reco object with parameters
reco = Recommender(data_path, factors = factors, regularization=reg, iterations=iterations)

print('start preperation...')

#load, clean and prepare data
reco.to_matrix()

print('sparse matrix ok')

#prepare matrix for training and dictionary to calculate hit rate
reco.make_train_2()

#save the prepped data
print('saving data..')
os.makedirs(save_folder, exist_ok=True)
train_path = os.path.join(save_folder, 'train.npz')
user_item_path = os.path.join(save_folder, 'user_item_altered.pkl')
object_path = os.path.join(save_folder, 'reco.pkl')

#save matrix and dictionnary
sparse.save_npz(train_path, reco.train)
joblib.dump(reco.user_item_altered, user_item_path)

#save object reco
joblib.dump(reco, object_path)

run.complete()
