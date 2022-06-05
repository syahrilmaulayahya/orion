# Importing essential libraries

from flask import Flask,render_template,url_for,request
import pandas as pd 
import os
import pickle
import numpy as np

# Load the model
model = 'finalized_model.pkl'
rf_classifier = pickle.load(open(model, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

def categoryClass(data):
    classes = rf_classifier.predict(data)
    if classes == 0:
        return 'GALAXY'
    elif classes == 1:
        return 'STAR'
    elif classes == 2:
        return 'QSO'


@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        alpha = float(request.form['alpha'])
        delta = float(request.form['delta'])
        ultraviolet = float(request.form['u'])
        green = float(request.form['g'])
        red = float(request.form['r'])
        near_infraret = float(request.form['i'])
        infraret = float(request.form['z'])
        redshift = float(request.form['redshift'])
        
        data = np.array([[alpha,delta, ultraviolet, green, red, near_infraret, infraret, redshift]])
        my_prediction = categoryClass(data)
    
        return render_template('index.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)