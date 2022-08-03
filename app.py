from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import pickle
import string
STOPWORDS = set(stopwords.words('english'))
stemmer = nltk.SnowballStemmer("english")

app = Flask(__name__)

@app.route('/',methods = ["GET","POST"])
@app.route('/home', methods = ["GET","POST"])

def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def load_page():
    tweet = ''
    if request.method == 'POST':
        tweet = request.form['inputTweet']
        result = prediction(tweet)
        print(type(result))
        result = "The tweet represents " + result
        print(result)
        result = str(result)
        return render_template('index.html', result=str(result))
    
def prediction(text):
    clean_text = text_preprocessing(text)
    cv = pickle.load(open('cv_model.sav', 'rb'))
    test = cv.transform([clean_text]).toarray()
    dt_cv = pickle.load(open('dt_model.sav', 'rb'))
    return dt_cv.predict(test)

def remove_stopwords(text):
    return ' '.join(word for word in str(text).split() if word not in STOPWORDS)

def stemming(text):
    return ' '.join(stemmer.stem(word) for word in str(text).split())
    
def text_preprocessing(sen):
    sen = str(sen).lower()
    sen = re.sub('[^a-zA-Z]', ' ', sen)  # remove punctuations and numbers
    sen = re.sub(r'\s+', ' ', sen) # remove extra spaces from the data
    sen = remove_stopwords(sen) # remove stopwords
    sen = stemming(sen) # apply stemming
    return sen

if __name__ == '__main__':  
    app.run(debug=True)