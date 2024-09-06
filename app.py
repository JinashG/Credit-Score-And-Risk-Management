import numpy as np
import loan_amount as lm
import pandas as pd
from flask import Flask, redirect, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/page2')
def page2():
    return render_template('page2.html')

@app.route('/page3')
def page3():
    return render_template('page3.html')

@app.route('/page4')
def page4():
    return render_template('page4.html')

@app.route('/submit', methods=['GET','POST'])
def submit():
    if request.method=="POST":
        accounts =float(request.form.get('acc'))
        education = float(request.form.get('edu'))
        employ=float(request.form.get('employment'))
        payhis=float(request.form.get('his'))
        balance=float(request.form.get('bal'))
        lim=float(request.form.get('limit'))
        credit_utilization_ratio=balance/lim
        credit_score = ((payhis//83.42)*0.35)+(credit_utilization_ratio*0.30)+(accounts*0.15)+(education*0.10)+(employ* 0.10)
        if(credit_score>900):
            credit_score=900
        if(credit_score<250):
            credit_score=250
        return render_template('page2.html', value=round(credit_score))
# Define other routes and backend logic as needed

@app.route('/predict', methods=['POST'])
def predict():
    return lm.predict_amount()

if __name__ == '__main__':
    app.run(debug=True)