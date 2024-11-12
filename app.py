import re
import gensim
import joblib
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, render_template

import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# To average the vectors of each word in a sentence to a single vector
def avg_word2vec(doc):
    # remove out-of-vocabulary words
    w2v_model = gensim.models.Word2Vec.load("./models/word2vec.model")
    return np.mean([w2v_model.wv[word] for word in doc if word in w2v_model.wv.index_to_key],axis=0)

def make_prediction (issue):
    lemmatizer=WordNetLemmatizer()

    i = re.sub('[^a-zA-Z?!]', ' ', issue)
    i = i.lower()
    i = i.split()
    i = [lemmatizer.lemmatize(word) for word in i if not word in stopwords.words('english')]
    i = [' '.join(i)]
    x=[avg_word2vec(word) for word in i]
    x = np.asarray(x, dtype="object")
    # the input to the model should be of the shape (1,100)
    x.reshape(1,-1)
    # load the random forest classifier
    classifier = joblib.load("./models/rf_classifier.joblib")

    return classifier.predict_proba(x).tolist()[0]

@app.route('/api/predict',methods=["POST"])
def predict():
    if request.method == 'POST':
        issue_title=request.args['title']
        issue_body=request.args['body']
    bug,enhancement,question=make_prediction(f'{issue_title} {issue_body}')
    return f'Probability of bug: {bug:.2f} \nProbability of enhancement: {enhancement:.2f}\nProbability of question: {question:.2f}'

@app.route('/api/correct', methods=["POST"])
def correct():
    if request.method == 'POST':
        issue_id=request.args['id']
        corrected_label=request.args['label']
    return f'Issue ID: {issue_id}'
    
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)