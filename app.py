from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load your model
try:
    with open('./Model/movement_classification_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    print("Model loaded successfully.")
    print("Model type:", type(model))  # Debugging statement to check the model type
except FileNotFoundError:
    print("Model file not found. Please check the path.")
    model = None
except Exception as e:
    print(f"Error loading the model: {e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Get feature data (adjust keys if necessary)
    features = [data['accX'], data['accY'], data['accZ'],
                data['gyroX'], data['gyroY'], data['gyroZ']]
    
    # Convert to NumPy array and reshape for the model
    input_data = np.array(features).reshape(1, -1) 
    
    # Make the prediction
    prediction = model.predict(input_data)[0]
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(port=5040)
