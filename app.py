import os
import re
import gensim
import joblib
import numpy as np
from flask_cors import CORS
from langdetect import detect
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask import Flask,jsonify, request
from supabase import create_client, Client
import warnings
warnings.filterwarnings("ignore")

load_dotenv() 
url= os.environ['SUPABASE_URL']
key= os.environ['SUPABASE_KEY']

supabase: Client = create_client(url, key)

app = Flask(__name__)
CORS(app)

LABELS=['bug','enhancement','question']

def insert_into_db(issue_title,issue_body,predicted_label):
    '''
    Inserts the issue title, issue body and the predicted label into the database.
    Returns the id number for the issue
    '''
    # Get the current number of rows in the dataset, which is used to calculate the issue id
    response = (supabase.table("issues")
                        .select("*",count="exact")
                        .execute())
    issue_id=response.count+1
    #insert the id, title, body, predicted label of the issue into the database
    response = (supabase.table("issues")
                        .insert({"id":issue_id,"issue_title": issue_title,'issue_body':issue_body,'predicted_label':predicted_label})
                        .execute())
    return issue_id

def update_issue_in_db(issue_id,corrected_label):
    '''
    Inserts the corrected label for an issue in the database using its id
    '''
    # Check if an issue with the provided issue_id exists in the table 
    response1 = (supabase.table("issues")
                         .select("*",count="exact")
                         .eq("id", issue_id)
                         .execute())
    #if the issue exists then update the label, otherwise return an error message  
    if response1.count==1:
        response2 = (supabase.table("issues")
                            .update({"corrected_label": corrected_label})
                            .eq("id", issue_id)
                            .execute())
        predicted_label=LABELS[response1.data[0]['predicted_label']]
        return {"success":f"Label for issue id {issue_id} was changed from {predicted_label} to {LABELS[corrected_label]}"}
    else:
        return {'error':f'Issue with id {issue_id} does not exist'} 

def avg_word2vec(doc):
    '''
    Averages the vectors of each word in a sentence to a single vector
    '''
    # remove out-of-vocabulary words
    w2v_model = gensim.models.Word2Vec.load("./models/word2vec.model")
    return np.mean([w2v_model.wv[word] for word in doc if word in w2v_model.wv.index_to_key],axis=0)

def make_prediction (issue):
    '''
    Pre-processes the text and makes a prediction using the trained model
    Returns a list of the predicted probabilities of the labels  
    '''
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
    '''
    API endpoint to predict label (bug, enhancement, question) from issue title and body
    '''
    if request.method == 'POST':
        issue_title=request.form.get('title')
        issue_body=request.form.get('body')
        # Check if the title and body are empty 
        if issue_title=="":
            return jsonify({'error':'Please enter the issue title'})
        if issue_body=="":
            return jsonify({'error':'Please enter the issue body'})
        # Check if the title and body are in English 
        if detect(issue_title)!='en':
            return jsonify({'error':'Please enter the issue title in English'})
        if detect(issue_body)!='en':
            return jsonify({'error':'Please enter the issue body in English'})
        
        preds=make_prediction(f'{issue_title} {issue_body}')
        max_prob_label=preds.index(max(preds))
        probs=[round(pred,2) for pred in preds]
        issue_id=insert_into_db(issue_title,issue_body,max_prob_label)
        issue_label=LABELS[max_prob_label]
        issue = {'id':issue_id,'label': issue_label,'probs':probs}
        return jsonify(issue)

@app.route('/api/correct', methods=["POST"])
def correct():
    '''
    API endpoint to correct a predicted label
    '''
    if request.method == 'POST':
        issue_id=str(request.form.get('issue_id'))
        corrected_label=str(request.form.get('corrected_label')).lower()
        
        if issue_id=="":
            return jsonify({'error':'Please enter an issue id'})
        if not issue_id.isnumeric():    
            return jsonify({'error':'Please enter a valid numeric issue id'})
        if corrected_label=="" or corrected_label not in LABELS:
            return jsonify({'error':'Please enter a valid corrected label (bug,enhancement,question)'})

    return jsonify(update_issue_in_db(int(issue_id),LABELS.index(corrected_label)))

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)