
print('print importing lib...')

import argparse
from azureml.core import Run
from azureml.core import Dataset
import joblib
import os

import numpy as np
import pandas as pd

from scipy import sparse
import random
import implicit
import joblib
import glob

print('lib imported...')

#get scripts arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input-data', type=str, dest='dataset_folder')
#outfolder for preprocessed data to use in pipeline. see OutputFileDatasetConfig doc from azure
parser.add_argument('--prepped_data', type=str, dest='prepped_data')
args = parser.parse_args()

save_folder = args.prepped_data

#set parameters


#get the experiment run context and workspace
run = Run.get_context()
ws = run.experiment.workspace

#load data
print('loading data...')
#frame = Dataset.get_by_name(ws, dataset_name).to_pandas_dataframe()
# Get the training data path from the input. must be passed as argument in the script as as_named_input + as_download or as_mount
data_path = run.input_datasets['clicks']
#import in dataframe
all_files = glob.glob(data_path + "/*.csv")
frame = pd.concat((pd.read_csv(f) for f in all_files), axis=0, ignore_index=True)
print('data loaded...')

################################################################################################################################
#                FUNCTION DEFINITION
################################################################################################################################

def make_train_2(ratings, pct_test = 0.2):
    '''
    Function take original user_article matrix choose a percentage of random user and mask one article 
    per user selected. it returns the original matrix, the new matrix and a dictionary of the masked pair
    user article
    
    '''
     # Make a copy of the original set to be the test set.
    test_set = ratings.copy() 
    # Store the test set as a binary preference matrix
    test_set[test_set != 0] = 1 
    # Make a copy of the original data we can alter as our training set.
    training_set = ratings.copy()  
    
    # Find the indices in the ratings data where an interaction exists
    nonzero_inds = training_set.nonzero() 
    # Zip these pairs together of user,item index into list
    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1])) 
    
    
    random.seed(0) 
    # Round the number of samples needed to the nearest integer 
    num_samples = int(np.ceil(pct_test*training_set.shape[0])) 
    # Sample a random number of user without replacement
    sample_user = random.sample(set(list(nonzero_inds[0])), num_samples)
    
    # selec one random article per user
    item_ind=[]
    for user in sample_user:
        list_artic_user = [index[1] for index in nonzero_pairs if index[0]==user]
        article_hide = random.sample(list_artic_user, 1)
        item_ind.extend(article_hide) 
    
    # Assign all of the randomly chosen user-item pairs to zero
    training_set[sample_user, item_ind] = 0 
    # Get rid of zeros in sparse array storage after update to save space
    training_set.eliminate_zeros()
    
    #dictionary of pairs
    user_item_hide = dict(zip(sample_user, item_ind))
    
    # Output the unique list of user rows that were altered  
    return training_set, test_set, user_item_hide

#############################################################################################################
#                DATA PREP
#############################################################################################################
print('start preperation...')
#keep column of interest
#after extracting from csv dtype = object with must be turn into int or float for csr matrix
user_article = frame[['user_id', 'click_article_id']]


#make sparse matric : np.ones_like to put the wieght at one (read or not read) could be rating if existed 
matrix_user_article = sparse.csc_matrix((np.ones_like(user_article['user_id'].astype(int)), (user_article['user_id'].astype(int), user_article['click_article_id'].astype(int))))
print('sparse matrix ok')

#make train test set
train, test, user_item_altered = make_train_2(matrix_user_article, pct_test = 0.2)

#save the prepped data
print('saving data..')
os.makedirs(save_folder, exist_ok=True)
train_path = os.path.join(save_folder, 'train.npz')
test_path = os.path.join(save_folder, 'test.npz')
user_item_path = os.path.join(save_folder, 'user_item_altered.pkl')

sparse.save_npz(train_path, train)
sparse.save_npz(test_path, test)
joblib.dump(user_item_altered, user_item_path)

run.complete()
