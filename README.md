# Projet8_recommender
## Content
The folder **model development** contains the notebook to run the pipeline in Azure (be sure to add your own *config.json* to connect to your workspace). The scripts are all in the folder training_azure. The notebook *P08_azure_pipeline_class* build the pipeline and call the scripts located in the folder. *P08_recommender_class* defines the class used in the project. 
  
The folder **application** contains the scripts to build the app with Flask along with the best model saved and the user_item matrix from the training.  
The app is an API endpoint which take a json object with the user ID number and return a list of 5 recommendations.

## Deploy the app
To deploy the app yourself using Heroku follow the steps below:  
* Step 1  
Clone the following repo which contains only the python scripts for the app:
`git clone <https://github.com/Falco-ops/p8_flask_recommender>`  
  
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
 `$ git commit -m "first commit"
 
 * Step 6  
 Deploy code.
 `$ git push heroku main`
 
 to visualize the build logs type:
 `$ heroku logs`
 
 * Step 7  
 Test the endpoint with python.
 The url endpoint will be given in git bash usualy in this form: `https://nameOfApp.heroku`
 don't forget to add *predict* or *test* at the end.
 ```python
 import requests
 url_test = 'https://nameOfApp.heroku/test'
 url_predict = 'https://nameOfApp.heroku/predict'
 payload = {userId: 14}
 
 #test ping
 r = requests.get(url)
 r.text
 
 #test prediction
 r =requests.post(url, json=payload)
 r.text
 ```
 
 
 
 
