import sys
import json
import joblib
import numpy as np


def main():
    # Read stdin fully and parse JSON; return a JSON error on failure.
    try:
        raw = sys.stdin.buffer.read()
        if not raw:
            print(json.dumps({'error': 'no stdin provided'}))
            return
        data = json.loads(raw.decode('utf-8'))
    except Exception as e:
        print(json.dumps({'error': f'failed to parse stdin JSON: {e}'}))
        return

    features = data.get('features')
    if features is None:
        print(json.dumps({'error': 'no features'}))
        return

    X = np.array(features, dtype=float).reshape(1, -1)

    try:
        model = joblib.load('model.sav')
    except Exception as e:
        print(json.dumps({'error': f'model load failed: {e}'}))
        return

    try:
        scaler = joblib.load('scaler.sav')
    except Exception:
        scaler = None

    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception:
            pass

    try:
        pred = model.predict(X)[0]
    except Exception as e:
        print(json.dumps({'error': f'predict failed: {e}'}))
        return

    try:
        probs = model.predict_proba(X)[0]
        confidence = float(max(probs) * 100)
    except Exception:
        confidence = 95.0

    print(json.dumps({'prediction': int(pred), 'confidence': confidence}))


if __name__ == '__main__':
    main()
