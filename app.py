import os
from flask import Flask, jsonify, request
from predict_live import predict_live
from live_api import GERMAN_SOLAR_FARMS

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'service': 'Solar Power Forecasting Live API',
        'status': 'active',
        'endpoints': {
            '/farms': 'GET - List all 21 solar farm locations',
            '/predict': 'GET - Predict solar power output for a plant. Query params: plant_id (1-21), model (ann|lstm)'
        }
    })

@app.route('/farms', methods=['GET'])
def get_farms():
    return jsonify({'count': len(GERMAN_SOLAR_FARMS), 'farms': GERMAN_SOLAR_FARMS})

@app.route('/predict', methods=['GET'])
def predict():
    try:
        plant_id_str = request.args.get('plant_id', '1')
        model_type = request.args.get('model', 'ann').lower()

        if not plant_id_str.isdigit() or not (1 <= int(plant_id_str) <= 21):
            return jsonify({'error': 'plant_id must be an integer between 1 and 21'}), 400

        if model_type not in ['ann', 'lstm']:
            return jsonify({'error': "model must be either 'ann' or 'lstm'"}), 400

        plant_id = int(plant_id_str)
        result = predict_live(plant_id=plant_id, model_type=model_type)

        return jsonify({'success': True, 'data': result})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
