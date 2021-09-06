
print('print importing lib...')

import argparse
from azureml.core import Run, Model
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
parser.add_argument("--training-data", type=str, dest='training_data', help='training data')
args = parser.parse_args()

save_folder = args.training_data

# Get the experiment run context
run = Run.get_context()

# load the prepared data file in the training folder
print('loading data..')

train_path = os.path.join(save_folder, 'train.npz')
test_path = os.path.join(save_folder, 'test.npz')
user_item_path = os.path.join(save_folder, 'user_item_altered.pkl')

train = sparse.load_npz(train_path)
test = sparse.load_npz(test_path)
user_item_altered = joblib.load(user_item_path)

print('data loaded..')

##################################################################################################
#                FUNCTION DEFINITION
##################################################################################################
def predict_evaluate(model, train, user_item_altered):
    '''
    This function make 10 predictions for every users that had an article hiden during the make train process.
    It then evaluate if the hidden article is included in the 10 predictions.
    It calculate the regular hit nbr_of_hit / nbr_of_user_altered.
    And it calculates the second matrix which take into account the position of the hidden artcile in the 
    prediction list. 
    
    Metrics : hit rate and average reciprocal hit rank
    
    '''
    
    hit = 0
    sum_rev_pos = 0
    for key, value in user_item_altered.items():
        #make recommendation for each altered user
        recommendation = model.recommend(key, train)
        #store in list
        recommended_item = [index[0] for index in recommendation]

        #check if hiden article is in the recommendation list. calculate hit rate (HR) and average reciprocal
        #hit rank (ARHR)

        if user_item_altered[key] in recommended_item:
            #number of hit
            hit+=1
            
            #get positon of the hit in the recommendation list
            pos = recommended_item.index(user_item_altered[key])+1
            sum_rev_pos = sum_rev_pos+(1/pos)

    #hit rate
    HR = hit/(len(user_item_altered))

    #average reciprocal hit rank
    f = 1/len(user_item_altered)
    ARHR = f*sum_rev_pos

    return HR, ARHR

##################################################################################################
#                TRAINING
##################################################################################################

#Build model from implicit librairy
# factors : numbers of laten factors to compute
# regularization : reg factor
#iterations : number of ALS iteraction to use when fitting data
model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=20)

#to fit the model we need the transpose from the train matrix i.e article_user
train_T = train.transpose()

print("Fitting..")
#fit model on csr matrix
model.fit(train_T)

print('Evaluate..')
#evaluate with the original csr matrix 
HR, ARHR = predict_evaluate(model, train, user_item_altered)

print("Hit Rate of:", HR)
print("Average Reciprocal Hit Rate of:", ARHR)
run.log('HR', np.float(HR))
run.log('ARHR', np.float(ARHR))

#save the trained model in the outputs folder
print("saving model..")
os.makedirs('outputs', exist_ok=True)
model_file = os.path.join('outputs', 'recommender_model.pkl')
joblib.dump(value=model, filename=model_file)

#register the model
Model.register(workspace=run.experiment.workspace,
              model_path = model_file,
              model_name = 'recommender_model',
              properties={'Hit Rate': np.float(HR),
                         "Average Reciprocal Hit Rate": np.float(ARHR)})

run.complete()
