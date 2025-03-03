from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load Model and Label Encoders
with open('Statistics\\Project\\AWS Deployment\\App\\coping_mechanism_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('Statistics\\Project\\AWS Deployment\\App\\label_encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_copingMechanism():
    age = request.form.get('age')
    gender = request.form.get('gender')
    residence = request.form.get('residence')

    # Encode Input Using Stored LabelEncoders
    try:
        age = label_encoders['Age'].transform([age])[0]
        gender = label_encoders['Gender'].transform([gender])[0]
        residence = label_encoders['ResidenceType'].transform([residence])[0]
    except ValueError:
        return "Invalid input values!"

    # Convert to NumPy Array
    input_data = np.array([[age, gender, residence]])

    # Make Prediction
    prediction = model.predict(input_data)[0]

    return f"Predicted Coping Mechanism: {prediction}"

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080)
