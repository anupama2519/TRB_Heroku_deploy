# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 22:45:10 2020

@author: admin
"""
import numpy as np
import pandas as pd
from flask import Flask,render_template,request,jsonify
import pickle


app = Flask(__name__)
model =pickle.load(open('finalmodel_TRB.pkl','rb'))
cols_when_model_builds = model.get_booster().feature_names

@app.route('/') 
def home(): 
   return render_template('temp_index.html',prediction_test=0) 

@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == "POST":
        #int_features=[float(x) for x in request.form.values()]
        loan_amnt = request.form.getlist('loan_amnt ')
        Rate_of_intrst = request.form.getlist('Rate_of_intrst')
        annual_inc = request.form.getlist('annual_inc')
        debt_income_ratio = request.form.getlist('debt_income_ratio')
        numb_credit = request.form.getlist('numb_credit')
        total_credits = request.form.getlist('total_credits')
        total_rec_int = request.form.getlist('total_rec_int')
        tot_curr_bal = request.form.getlist('tot_curr_bal')
        new_row ={'loan_amnt ':int(loan_amnt[0]),'Rate_of_intrst':float(Rate_of_intrst[0]),
                  'annual_inc':float(annual_inc[0]),
                  'debt_income_ratio':float(debt_income_ratio[0]),'numb_credit':int(numb_credit[0]),
                  'total_credits':int(total_credits[0]),'total_rec_int':float(total_rec_int[0]),
                  'tot_curr_bal':int(tot_curr_bal[0])}
        
        print(new_row)
        final_features = pd.DataFrame()
        final_features=final_features.append(new_row,ignore_index=True)
        final_features=final_features[cols_when_model_builds]
        prediction = model.predict(final_features)
        output = round(prediction[0],2)
        return render_template('temp_index.html',prediction_test=format(output))


if __name__ == "__main__":
  app.run(debug=True)
  