
print('print importing lib...')

import argparse
from azureml.core import Run, Model
from azureml.core import Dataset
from scipy import sparse
import joblib
import os
import numpy as np
import pandas as pd
import joblib
from class_recommender import Recommender


print('lib imported...')

#get scripts arguments
parser = argparse.ArgumentParser()
parser.add_argument("--training-data", type=str, dest='training_data', help='training data')
args = parser.parse_args()

save_folder = args.training_data

# Get the experiment run context
run = Run.get_context()

# load the prepared data file in the training folder
print('loading object..')

object_path = os.path.join(save_folder, 'reco.pkl')


reco = joblib.load(object_path)

print('object loaded..')

#fit model 
reco.fit()

print('fitted..')

#evaluate 
HR, ARHR = reco.predict_evaluate()

print("Hit Rate of:", HR)
print("Average Reciprocal Hit Rate of:", ARHR)
run.log('HR', np.float(HR))
run.log('ARHR', np.float(ARHR))

#save the trained model in the outputs folder
print("saving model..")
os.makedirs('outputs', exist_ok=True)
model_file = os.path.join('outputs', 'recommender_model.pkl')
joblib.dump(value=reco.model, filename=model_file)

#register the model
Model.register(workspace=run.experiment.workspace,
              model_path = model_file,
              model_name = 'recommender_model',
              properties={'Hit Rate': np.float(HR),
                         "Average Reciprocal Hit Rate": np.float(ARHR)})

run.complete()
