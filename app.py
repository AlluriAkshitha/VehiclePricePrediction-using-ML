from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load your trained model
model = pickle.load(open('vehicle_price_xgb_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            make = float(request.form.get('make'))
            model_f = float(request.form.get('model'))
            vehicle_age = float(request.form.get('vehicle_age'))
            mileage = float(request.form.get('mileage'))
            cylinders = float(request.form.get('cylinders'))
            doors = float(request.form.get('doors'))
            engine_type = float(request.form.get('engine_type'))
            fuel_gasoline = 1 if request.form.get('fuel_gasoline') == 'on' else 0
            transmission_automatic = 1 if request.form.get('transmission_automatic') == 'on' else 0
            drivetrain_fwd = 1 if request.form.get('drivetrain_fwd') == 'on' else 0
            body_sedan = 1 if request.form.get('body_sedan') == 'on' else 0

            features = np.array([[make, model_f, vehicle_age, mileage, cylinders, doors, engine_type,
                                  fuel_gasoline, transmission_automatic, drivetrain_fwd, body_sedan]])
            result = model.predict(features)

            # Handle both array-like and scalar (float) outputs
            if isinstance(result, (np.ndarray, list, tuple)):
                prediction = float(result[0])
            else:
                prediction = float(result)

            return render_template('result.html', prediction=round(prediction, 2), is_error=False)
        except Exception as e:
            error_message = f"Error: {e}"
            return render_template('result.html', prediction=error_message, is_error=True)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
