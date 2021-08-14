# import Flask class from the flask module
from flask import Flask, request, jsonify, render_template
import pandas as pd

import numpy as np
import pickle

user_final_rating = pd.read_pickle("models/user_final_rating.pkl")

data = pd.read_csv("models/data.csv")

word_vectorizer = pickle.load(open('models/word_vectorizer.pkl', 'rb'))
classifier_logit = pickle.load(open('models/logit_model.pkl', 'rb'))


def recommend(user_input):
    d = user_final_rating.loc[user_input].sort_values(ascending=False)[0:20]

    # Based on positive sentiment percentage.
    i = 0
    a = {}
    for prod_name in d.index.tolist():
        product_name = prod_name
        product_name_review_list =data[data['name'] == product_name]['final_review'].tolist()
        features= word_vectorizer.transform(product_name_review_list)
        classifier_logit.predict(features)
        a[product_name] = classifier_logit.predict(features).mean()*100

    b= pd.Series(a).sort_values(ascending = False).head(5).index.tolist()
    return b


# Create Flask object to run
app = Flask(__name__)


# Home page
@app.route('/')
def home():
    return render_template('Product_Recommendation.html')


# Prediction page
@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    username = str(request.form.get('reviews_username'))
    print(username)
    prediction = recommend(username)
    print("Output :", prediction)

    return render_template('Product_Recommendation.html', message='Your Top 5 Product Recommendations are:\n ', username = username,results = prediction)


if __name__ == "__main__":
    print("**Starting Server...")
    # Call function that loads Model
    print("**Model loaded...")
    # Run Server
    app.debug = True
    app.run()#(host="0.0.0.0", port=5000)
    # app.run(debug = True)
