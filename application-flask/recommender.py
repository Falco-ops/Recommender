import implicit
import joblib
from scipy import sparse
import os
import sys

dirname = os.path.dirname(__file__)
name = 'recommender_model.pkl' 
filename = os.path.join(dirname, name)
#model
model = joblib.load(filename)
#matrix
matrix = 'matrix_user_article.npz'
filename2 = os.path.join(dirname, matrix)
matrix_user_article = sparse.load_npz(filename2)

def make_recommendation(user):
    
    #make recommendation for each altered user
    recommendation = model.recommend(user, matrix_user_article, 5)
    #store in list
    recommended_item = [index[0] for index in recommendation]
    
    
    return recommended_item

if __name__ == '__main__':
    make_recommendation(sys.argv[1])
