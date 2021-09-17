# Projet8_recommender
## Content
The folder **model development** contains the notebook to run the pipeline in Azure (be sure to add your own *config.json* to connect to your workspace). The scripts are all in the folder training_azure. The notebook *P08_azure_pipeline_class* build the pipeline and call the scripts located in the folder. *P08_recommender_class* defines the class used in the project. 
  
The folder **application** contains the scripts to build the app with Flask along with the best model saved and the user_item matrix from the training.  
The app is an API endpoint which take a json object with the user ID number and return a list of 5 recommendations.

## Model development  
[Pipeline building notebook](https://github.com/Falco-ops/projet8_recommender/blob/main/model_devlopment/p08_azure_pipeline_Class.ipynb):  
Final notebook to prepare the data and train the model. Define environment, compute power, pipeline step and register the model to Azure ML.  

[Class_recommender script](https://github.com/Falco-ops/projet8_recommender/blob/main/model_devlopment/training_azure/class_recommender.py):  
Define the class named _Recommender_ with methods and attributes used in the project.  

[Data_prep_class script](https://github.com/Falco-ops/projet8_recommender/blob/main/model_devlopment/training_azure/data_prep_class.py):  
Script used for the first step of the pipeline to prepare the raw data.  
**Input**: data folder (Azure) and model hyperparameters (factors, iterations and regularization).  
**Output**: recommender object, matrix user_item use for training, dictionary user item pairs hidden for the test.

[Train_recommender_class scripts](https://github.com/Falco-ops/projet8_recommender/blob/main/model_devlopment/training_azure/train_recommender_class.py):    
Script used in the second step of the pipeline to train and register the model.  
**Input**: intermediary data folder containing output from step 1.  
**Output**: the trained model and the user_item matrix.

[Environment](https://github.com/Falco-ops/projet8_recommender/blob/main/model_devlopment/training_azure/env-p8.yml):  
.yml used for the environment by Azure.


## Deploy the app
To deploy the app yourself using Heroku follow the steps below:  
* Step 1  
Clone the following repo which contains only the python scripts for the app:  
```git clone https://github.com/Falco-ops/p8_flask_recommender```  
  
 * Step 2  
 Sign up to Heroku and download the Heroku CLI <https://www.heroku.com>
 
 * Step 3  
 Open Git Bash in the folder containing the repo and login to the Heroku CLI:  
 `$ heroku login`  
 This will open your browser and ask for your login info.
 
 * Step 4  
 Create a new app in Heroku.  
 `$ heroku create <nameOfApp>`  
  For an existing app you can use  
 `$ heroku git:remote -a <nameOfApp>`
 
 * Step 5  
 Initialize a local Git repository and commit the app code to it.  
 `$ git init`  
 `$ git add .`  
 `$ git commit -m "first commit"`
 
 * Step 6  
 Deploy code.    
 `$ git push heroku main`  
 to visualize the build logs type:  
 `$ heroku logs`
 
 * Step 7  
 Test the endpoint with python.
 The url endpoint will be given in git bash once the build is complete usualy in this form: `https://nameOfApp.herokuapp.com`
 don't forget to add *predict* or *test* at the end.
 ```python
 import requests
 url_test = 'https://nameOfApp.herokuapp.com/test'
 url_predict = 'https://nameOfApp.herokuapp.com/predict'
 payload = {userId: 14}
 
 #test ping
 r = requests.get(url_test)
 r.text
 
 #test prediction
 r =requests.post(url_predict, json=payload)
 r.text
 ```
 
 
 
 
