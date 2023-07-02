from flask import Flask, render_template,request
import joblib

model1=joblib.load('./models/reg.pkl')
model2=joblib.load('./models/clus.pkl')

app=Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html',result1="not yed inputed", result2="not yet inputed")

@app.route('/predict', methods=['POST'])
def predict():
	
		x1=request.form.get('GRE Score')
		x2=request.form.get('TOEFL Score')
		x3=request.form.get('University Rating')
		x4=request.form.get('SOP')
		x5=request.form.get('LOR')
		x6=request.form.get('CGPA')
		x7=request.form.get('Research')
		result1=model1.predict([[x1,x2,x3,x4,x5,x6,x7]])
		result2=model2.predict([[x1,x2,x3,x4,x5,x6,x7,result1]])
		res=['Top student','Aspirational Student','Average Student']
		return render_template('index.html', result1='{0:.2f}'.format(result1[0]*100), result2=res[result2[0]])

if __name__=='__main__':
	app.run(host="0.0.0.0", port=5000)

