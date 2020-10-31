import pandas as pd
import numpy as np
import json
from google_play_scraper import Sort, app, reviews

import warnings
warnings.filterwarnings('ignore')

def get_app_reviews():

    #Choose the app to obtain the reviews for
    app_package = ['com.medium.reader']
    app_reviews = []

    for app_name in app_package:
        #iterate through the reviews from 1-5
        for review_score in list(range(1,6)):
            #Choose the most relevant and the newest reviews first
            for sort_score_order in [Sort.MOST_RELEVANT]:
                app_review_list, _ = reviews(
                    app_name,
                    lang='en',
                    country='us',
                    sort=sort_score_order,
                    count=1500 if review_score==3 else 750,
                    filter_score_with=review_score
                )
                print(app_review_list)
    
            for review in app_review_list:
                review['sort_order'] = 'most_relevant' if sort_score_order == Sort.MOST_RELEVANT else 'newest'
                review['app_id'] = app_name

            app_reviews.extend(app_review_list)
            print(len(app_reviews))

    return app_reviews

def score_to_sentiment(rating):
    rating = int(rating)
    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    else: 
        return 2

def create_dataset(app_reviews_list):
    # Create Dataframe from app review list
    app_reviews_df = pd.DataFrame(app_reviews_list)
    #Filter out reviews that are less than 50 words
    #mask = app_reviews_df['content'].str.len() > 30
    #print(len(mask))
    #app_reviews_df = app_reviews_df[mask]
    temp_df = app_reviews_df[['content', 'score']]
    print(temp_df.score.value_counts())
    #temp_df.dropna(inplace=True)
    temp_df['sentiment'] = temp_df.score.apply(score_to_sentiment)
    temp_df.drop(columns=['score'], inplace=True)
    print(len(temp_df))
    print(temp_df.info())
    temp_df.to_csv('app_reviews.csv', index=None, header=True)

if __name__ == "__main__":

    app_reviews_list = get_app_reviews()
    create_dataset(app_reviews_list)

    
    
