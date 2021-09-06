
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
    data = json.loads(data)
    value = data["userId"]
    

    results = make_recommendation(value)

    
    return json.dumps(eval(str(results)))

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
