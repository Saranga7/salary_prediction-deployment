import numpy as np
import pickle
from flask import Flask, request,jsonify,render_template

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    features=[int(x) for x in request.form.values()]
    final_features=[np.array(features)]
    pred=model.predict(final_features)

    output=round(pred[0])

    return render_template('index.html',prediction_text='Employee Salary should be $ {}'.format(output))

if __name__=="__main__":
    app.run(debug=True)

