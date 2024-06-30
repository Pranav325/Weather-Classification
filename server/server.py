# from flask import Flask, request, jsonify
# import json 
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_squared_error
# import pickle
# from flask_cors import CORS
# app = Flask(__name__)
# CORS(app) 

# def make_predictions(model, data):
#     predictions = model.predict(data)
#     return predictions

# @app.route('/predict', methods=['POST'])
# def predict():
#     data_str = request.json.get('predict') 
#     weather_values = data_str.split(',')
#     feature_names = [ "Temperature","Humidity","Wind Speed","Precipitation (%)","Cloud Cover","Atmospheric Pressure","UV Index","Season", "Visibility (km)","Location"  
# ]
#     data = {feature_names[i]: int(weather_values[i]) for i in range(len(feature_names))}
#     data = pd.DataFrame.from_dict(data, orient='index').T
#     print(data)
#     with open('randomforest_model.pkl', 'rb') as model_file:
#         model = pickle.load(model_file)
#     predictions = make_predictions(model, data)
#     # predictions = 10 ** predictions  # Convert back to original scale
#     # predictions = int(predictions)
#     return {
#         'statusCode': 200,
#         'headers': {
#             'Access-Control-Allow-Headers': 'Content-Type',
#             'Access-Control-Allow-Origin': '*',
#             'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
#         },
#         'body': f"${predictions}"
#     }


# if __name__ == '__main__':
#     app.run(debug=True)




from flask import Flask, request, jsonify
import json
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model once at startup
with open('randomforest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def make_predictions(model, data):
    predictions = model.predict(data)
    return predictions

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data_str = request.json.get('predict')
        weather_values = data_str.split(',')
        feature_names = [
            "Temperature", "Humidity", "Wind Speed", "Precipitation (%)",
            "Cloud Cover", "Atmospheric Pressure", "UV Index", "Season",
            "Visibility (km)", "Location"
        ]
        
        # Define the correct data types
        data_types = {
            "Temperature": float, "Humidity": int, "Wind Speed": float, 
            "Precipitation (%)": float, "Cloud Cover": int, "Atmospheric Pressure": float,
            "UV Index": int, "Season": int, "Visibility (km)": float, "Location": int
        }
        
        # Convert input values to the correct data types
        data = {feature_names[i]: data_types[feature_names[i]](weather_values[i]) for i in range(len(feature_names))}
        data = pd.DataFrame(data, index=[0])
        
        predictions = make_predictions(model, data)
        # predictions = 10 ** predictions
        # predictions = int(predictions[0])
        
        response = {
            'statusCode': 200,
            'body': f"${predictions}"
        }
    except Exception as e:
        response = {
            'statusCode': 500,
            'body': str(e)
        }
        
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
