from flask import Flask, render_template, request

application = Flask(__name__)
app = application

import pickle
model = pickle.load(open('models/model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def index():
      return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
      if request.method == 'POST':
            P=int(request.form.get('Pregnancies'))
            G=int(request.form.get('Glucose'))
            B=int(request.form.get('BloodPressure'))
            S=int(request.form.get('SkinThickness'))
            I=float(request.form.get('Insulin'))
            BMI=float(request.form.get('BMI'))
            DPF=float(request.form.get('DiabetesPedigreeFunction'))
            Age=int(request.form.get('Age'))

            new_scaled_input = scaler.transform([[P, G, B, S, I, BMI, DPF, Age]])

            result = model.predict(new_scaled_input)

            if result[0] == 0:
                  result = 'Non Diabetic '
            else:
                  result = 'Diabetic '

            return render_template('predict.html', results=result)
      else:
            return render_template('predict.html')

if __name__ == "__main__":
      app.run(debug=True, host='0.0.0.0', port=5000)