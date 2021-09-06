
from scipy import sparse
import random
import implicit
import numpy as np
import pandas as pd
import glob


class Recommender:
    '''class to train recommender'''
    

    
    def __init__(self, data_path, factors=20, regularization=0.1, iterations=20):
        self.data_path = data_path
        self.model = implicit.als.AlternatingLeastSquares(factors=factors, regularization=regularization, iterations=iterations) 
    
    def to_matrix(self):
        all_files = glob.glob(self.data_path + "/*.csv")        
        self.frame = pd.concat((pd.read_csv(f) for f in all_files), axis=0, ignore_index=True)
        self.frame = self.frame[['user_id', 'click_article_id']]
        self.user_article = sparse.csc_matrix((np.ones_like(self.frame['user_id'].astype(int)), 
                                               (self.frame['user_id'].astype(int), 
                                                self.frame['click_article_id'].astype(int))))
    
    def make_train_2(self, pct_test = 0.2):
        '''
        Function take original user_article matrix choose a percentage of random user and mask one article 
        per user selected. it returns the the new matrix and a dictionary of the masked pair
        user article

        '''
        # Make a copy of the original set to be the test set.
        #test_set = ratings.copy() 
        # next line in case of recommender with ratings
        #test_set[test_set != 0] = 1 
        # Make a copy of the original data we can alter as our training set.
        training_set = self.user_article.copy()  

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
        
        self.train = training_set
        self.user_item_altered = user_item_hide
        
    def fit(self):
        train_T = self.train.transpose()
        self.model.fit(train_T)

        


    def predict_evaluate(self):
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
        for key, value in self.user_item_altered.items():
            #make recommendation for each altered user
            recommendation = self.model.recommend(key, self.train)
            #store in list
            recommended_item = [index[0] for index in recommendation]

            #check if hiden article is in the recommendation list. calculate hit rate (HR) and average reciprocal
            #hit rank (ARHR)

            if self.user_item_altered[key] in recommended_item:
                #number of hit
                hit+=1

                #get positon of the hit in the recommendation list
                pos = recommended_item.index(self.user_item_altered[key])+1
                sum_rev_pos = sum_rev_pos+(1/pos)

        #hit rate
        HR = hit/(len(self.user_item_altered))

        #average reciprocal hit rank
        f = 1/len(self.user_item_altered)
        ARHR = f*sum_rev_pos

        return HR, ARHR


