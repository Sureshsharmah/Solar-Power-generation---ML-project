from flask import Flask, request, render_template
import joblib  # Assuming the model is saved using joblib
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load(r'D:\Projects\ML Projects\Solar Regression Pridiction\xgb_model.pkl')  # Adjust file name/path as necessary

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Capture form inputs
        distance_to_solar_noon = float(request.form['distance_to_solar_noon'])
        temperature = float(request.form['temperature'])
        wind_direction = float(request.form['wind_direction'])
        wind_speed = float(request.form['wind_speed'])
        sky_cover = float(request.form['sky_cover'])
        visibility = float(request.form['visibility'])
        humidity = float(request.form['humidity'])
        average_wind_speed = float(request.form['average_wind_speed'])
        average_pressure = float(request.form['average_pressure'])

        # Prepare inputs for prediction
        input_features = [[distance_to_solar_noon, temperature, wind_direction, wind_speed,
                           sky_cover, visibility, humidity, average_wind_speed, average_pressure]]

        # Make prediction
        prediction = model.predict(input_features)[0]

        return render_template('index.html', prediction_text=f"Predicted Power: {prediction:.2f} kW")
    except Exception as e:
        return render_template('index.html', error_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)