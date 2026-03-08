import os
import re
import gensim
import joblib
import numpy as np
from flask_cors import CORS
from langdetect import detect
from dotenv import load_dotenv
import nltk
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
from supabase import create_client, Client
from flask import Flask,jsonify, request,Response,render_template
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from prometheus_client import Counter, Gauge, generate_latest, Summary

import warnings
warnings.filterwarnings("ignore")
nltk.download('stopwords')

# Metrics initialization
accuracy_metric = Gauge('accuracy', 'Accuracy of predictions')
avg_confidence_metric = Gauge('avg_prediction_confidence', 'Average prediction confidence')
predictions_per_category = Counter('predictions_per_category', 'Number of predictions per category', ['label'])
correct_predictions = Counter('correct_predictions_per_category', 'Number of correct predictions per category', ['label'])
incorrect_predictions = Counter('incorrect_predictions_per_category', 'Number of incorrect predictions per category', ['label'])
request_latency = Summary('request_latency_seconds', 'Request latency in seconds')

# Track accuracy and confidence globally
total_predictions = Counter('total_predictions', 'Total number of predictions')
total_correct = Counter('total_correct', 'Total correct predictions')
confidence_sum = 0  # Total sum of confidence scores for average calculation

load_dotenv() 
url= os.environ['SUPABASE_URL']
key= os.environ['SUPABASE_KEY']

supabase: Client = create_client(url, key)

app = Flask(__name__)
CORS(app)

LABELS=['bug','enhancement','question']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/metrics')
def metrics():
    return Response(generate_latest(), mimetype='text/plain')

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
    #Load the model
    w2v_model = gensim.models.Word2Vec.load("./models/word2vec.model")
    # remove out-of-vocabulary words
    valid_words = [w2v_model.wv[word] for word in doc if word in w2v_model.wv.index_to_key]
    if not valid_words:
        return np.zeros(50)  # Return zero vector if no words are found
    return np.mean(valid_words, axis=0)

def filter_stopwords(text):
    # Noise Removal: converting every non alphabetical or non english character to blank space
    preprocessed_text = re.sub('[^a-zA-Z?!]', ' ', text)
    # Normalization: making every character lowercase
    preprocessed_text = preprocessed_text.lower()
    # Tokenization: splitting words by space
    preprocessed_text = preprocessed_text.split()
    preprocessed_text = [word for word in preprocessed_text if not word in stopwords.words('english')]
    return preprocessed_text

def make_prediction (issue_title,issue_body):
    '''
    Pre-processes the text and makes a prediction using the trained model
    Returns a list of the predicted probabilities of the labels  
    '''
    # filter and normalize stopwords
    preprocessed_title=filter_stopwords(issue_title)
    preprocessed_body=filter_stopwords(issue_body)

    lemmatizer=WordNetLemmatizer()
    # perform lemmatization 
    preprocessed_title = [lemmatizer.lemmatize(word) for word in preprocessed_title]
    preprocessed_body = [lemmatizer.lemmatize(word) for word in preprocessed_body]    

    # Compute separate vectors for title and body
    title_vectors=[avg_word2vec(preprocessed_title)]
    body_vectors=[avg_word2vec(preprocessed_body)]

    # Convert to NumPy arrays
    title_vectors = np.array(title_vectors)
    body_vectors = np.array(body_vectors)

    # Concatenate title and body vectors to form final feature vector
    feature_vec = np.hstack((title_vectors, body_vectors))

    # the input to the model should be of the shape (1,100)
    feature_vec.reshape(1,-1)

    # load the random forest classifier
    classifier = joblib.load("./models/rf_classifier.joblib")

    return classifier.predict_proba(feature_vec).tolist()[0]

@app.route('/api/predict',methods=["POST"])
@request_latency.time()
def predict():
    '''
    API endpoint to predict label (bug, enhancement, question) from issue title and body
    '''
    global confidence_sum

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
        
        # Check if the issue title and body only contain stopwords or not
        if filter_stopwords(issue_title)==[]:
            return jsonify({'error':'Issue title only contains stopwords'})
        if filter_stopwords(issue_body)==[]:
            return jsonify({'error':'Issue body only contains stopwords'})
        
        # Check the length of the title and body
        if len(issue_title)<10:
            return jsonify({'error':'Please enter a longer issue title'})
        if len(issue_body)<20:
            return jsonify({'error':'Please enter a longer issue body'})
        # If the length of the title or the body exceeds a maximum limit, truncate them
        if len(issue_title)>200:
            issue_title=issue_title[:200]
        if len(issue_body)>5000:
            issue_body=issue_body[:5000]
        
        # Initialize a thread pool
        executor = ThreadPoolExecutor(max_workers=2)
        # Submit the prediction task to the thread pool
        future = executor.submit(make_prediction, issue_title,issue_body)
        # If the prediction time exceeds a certain threshold we return an error
        PREDICTION_TIMEOUT = 20 #20 seconds
        # If the maximum probability is below a certain threshold we show a warning to the user
        CONFIDENCE_THRESHOLD=0.7
        try:
            # preds=make_prediction(f'{issue_title} {issue_body}')
            preds = future.result(timeout=PREDICTION_TIMEOUT)
            max_prob=max(preds)
            max_prob_label=preds.index(max_prob)
            probs=[round(pred,2) for pred in preds]
            
            # Update metrics
            confidence_sum += max(probs)
            total_predictions.inc()
            avg_confidence_metric.set(confidence_sum / total_predictions._value.get())
            predictions_per_category.labels(LABELS[max_prob_label]).inc()
            
            issue_id=insert_into_db(issue_title,issue_body,max_prob_label)
            issue_label=LABELS[max_prob_label]
            warn=''
            if max_prob<CONFIDENCE_THRESHOLD:
                warn=f'Warning: The prediction confidence is low. Please verify the predicted label.'
            issue = {'id':issue_id,'label': issue_label,'probs':probs,'warning':warn}
            return jsonify(issue)
        except TimeoutError:
            # Handle timeout by returning an error message
            return jsonify({"error": "Prediction timed out. Please try again later."})

@app.route('/api/correct', methods=["POST"])
@request_latency.time()
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

    result=update_issue_in_db(int(issue_id),LABELS.index(corrected_label))
    
    if "success" in result:
        # the message is: #f"Label for issue id {issue_id} was changed from {predicted_label} to {LABELS[corrected_label]}"
        predicted_label = result["success"].split(' ')[8]
        if corrected_label == predicted_label:
            total_correct.inc()
            correct_predictions.labels(corrected_label).inc()
        else:
            incorrect_predictions.labels(predicted_label).inc()

        accuracy_metric.set(total_correct._value.get() / total_predictions._value.get())

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)