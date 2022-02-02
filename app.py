
import numpy as np
import pickle
import math
from flask import Flask, render_template, request

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    
    loan_gender = request.form.get('gender')
    loan_married = request.form.get('married')
    loan_education = request.form.get('education')
    loan_selfEmployed = request.form.get('self_employed')
    
    loan_creditHistory = request.form.get('credit_History')
    loan_propertyArea = request.form.get('property_area')
    
    loan_applicantIncome = math.log(float(request.form.get('applicantIncome'))+2)
    loan_coapplicantIncome = request.form.get('coapplicantIncome')
    loan_loanAmount = math.log(float(request.form.get('loan_Amount'))+2)
    loan_loanAmountTerm = request.form.get('loan_Amount_Term')
    
    
    int_features = [loan_creditHistory, loan_applicantIncome, loan_loanAmount, 
                    loan_coapplicantIncome, loan_propertyArea, loan_loanAmountTerm, 
                    loan_selfEmployed, loan_education, loan_gender, loan_married]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    
    #prediction = model.predict([[1, 5849, 128, 0, 2, 360, 0, 0, 1, 0]])
    
    output = round(prediction[0],2)
    #print('output', output)
    if(output==1):
        output = 'Loan approved'
    else:
        output = 'Loan not approved'
        
    return render_template('index.html', prediction_text = f'{output}' 
                           #applicant_income=f'Applicant income: {loan_applicantIncome}',
                           #loan_amount=f'Loan amount: {loan_loanAmount}',
                           #credit_history=f'Credit history: {loan_creditHistory}'
                           )

if __name__ == '__main__':
    app.run(debug=True)