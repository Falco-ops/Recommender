
from flask import Flask, request, jsonify
from recommender import make_recommendation
import json

app = Flask('app')

@app.route('/test', methods=['GET'])
def test():
    return 'ping ok'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    value = data["userId"]
    

    results = make_recommendation(value)

    
    return json.dumps(eval(str(results)))


