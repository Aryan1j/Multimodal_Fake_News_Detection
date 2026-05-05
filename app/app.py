import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, render_template, request, jsonify

from src.hybrid_detector import predict_hybrid, predict_ensemble


app = Flask(__name__)


AVAILABLE_MODELS = {
    'bert':       'BERT',
    'roberta':    'RoBERTa',
    'distilbert': 'DistilBERT',
    'ensemble':   'Ensemble (All Models)',
}


@app.route('/')
def index():
    """Serve the main prediction interface."""
    return render_template('index.html', models=AVAILABLE_MODELS)


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        text      = data.get('text', '').strip()
        model_key = data.get('model', 'bert').lower()

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        if model_key not in AVAILABLE_MODELS:
            return jsonify({'error': f'Choose from: {list(AVAILABLE_MODELS.keys())}'}), 400

        # Route ensemble separately
        if model_key == 'ensemble':
            result = predict_ensemble(text)
        else:
            result = predict_hybrid(text, model_key)

        response = {
            'prediction':        result.label,
            'confidence':        result.confidence,
            'model':             result.model_display_name,
            'breakdown':         result.breakdown,
            'explanation':       result.explanation,
            'evidence_snippets': result.evidence_snippets,
        }

        if result.note:
            response['note'] = result.note

        return jsonify(response)

    except FileNotFoundError:
        return jsonify({'error': 'Model not found. Run: python -m src.train'}), 500
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/models', methods=['GET'])
def list_models():
    """Return list of available models."""
    return jsonify(AVAILABLE_MODELS)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})


def main():
    """Run the Flask development server."""
    print("="*60)
    print("Fake News Classifier - Web Interface")
    print("="*60)
    print("\nStarting server at http://localhost:5000")
    print("Press Ctrl+C to stop\n")

    app.run(host='0.0.0.0', port=5000, debug=True)


if __name__ == '__main__':
    main()