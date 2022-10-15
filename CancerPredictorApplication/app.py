import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
data = pd.read_csv('dataR2.csv')
X = data[['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 'Leptin', 'Adiponectin',
       'Resistin', 'MCP.1']]
y = data['Classification']
# scaler = MinMaxScaler()
# scaler.fit(X)
# scaler.transform(X)
model = DecisionTreeClassifier()
model.fit(X, y)
print(model.score(X, y))
# print(data.columns)
print(X.shape)
print(model.predict(X))
from flask import Flask, render_template, request
app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def basic():
    if request.method == 'POST':
        age = request.form['age']
        bmi = request.form['bmi']
        glucose = request.form['glucose']
        insulin = request.form['insulin']
        homa = request.form['homa']
        leptin = request.form['Leptin']
        adiponectin = request.form['Adiponectin']
        resistin = request.form['Resistin']
        mcp1 = request.form['mcp1']
        y_pred = [[age, bmi, glucose, insulin, homa, leptin, adiponectin, resistin, mcp1]]
        prediction_outcome = model.predict(y_pred)
        patient = "Sorry to say, you're suffering from Breast Cancer"
        success = " You  don't  seem  to  have  Breast  Cancer !!!"
        if prediction_outcome == 2:
            return render_template('index.html', p=patient)
        elif prediction_outcome == 1:
            return render_template('index.html', s=success)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)