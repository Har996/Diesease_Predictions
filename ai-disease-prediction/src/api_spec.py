"""OpenAPI 3 description of the REST API (served at /api/openapi.json, rendered at /api/docs)."""

_error = {"description": "Error", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}}

OPENAPI_SPEC = {
    "openapi": "3.0.3",
    "info": {
        "title": "AI Disease Prediction API",
        "version": "1.0.0",
        "description": ("Educational demo. Predictions are statistical estimates from a machine-learning model "
                        "and are NOT medical diagnoses. Consult a qualified healthcare professional."),
    },
    "paths": {
        "/predict": {"post": {
            "summary": "Predict the most likely disease from symptoms",
            "requestBody": {"required": True, "content": {"application/json": {
                "schema": {"$ref": "#/components/schemas/PredictRequest"},
                "example": {"symptoms": ["fever", "cough", "fatigue"], "age": 34, "gender": "female"}}}},
            "responses": {
                "200": {"description": "Prediction", "content": {"application/json": {
                    "schema": {"$ref": "#/components/schemas/PredictResponse"}}}},
                "400": _error, "413": _error, "503": {**_error, "description": "Model not trained yet"}}}},
        "/api/symptoms": {"get": {"summary": "List the symptoms the model understands",
                                  "responses": {"200": {"description": "Symptom list"}}}},
        "/api/model": {"get": {"summary": "Model metadata (name, version, class and feature counts)",
                               "responses": {"200": {"description": "Model info"}, "503": _error}}},
        "/health": {"get": {"summary": "Liveness / readiness check", "responses": {"200": {"description": "OK"}}}},
    },
    "components": {"schemas": {
        "PredictRequest": {"type": "object", "required": ["symptoms"], "properties": {
            "symptoms": {"type": "array", "items": {"type": "string"}, "minItems": 1,
                         "description": "Symptom identifiers from /api/symptoms (case-insensitive)."},
            "age": {"type": "number", "minimum": 0, "maximum": 120, "description": "Optional."},
            "gender": {"type": "string", "enum": ["female", "male", "other"], "description": "Optional."},
            "save": {"type": "boolean", "default": True,
                     "description": "Store the result in the caller's prediction history (cookie-based)."}}},
        "PredictResponse": {"type": "object", "properties": {
            "prediction": {"type": "string"},
            "confidence": {"type": "number", "description": "Model probability estimate (0-1), not a certainty."},
            "top_predictions": {"type": "array", "items": {"type": "object", "properties": {
                "disease": {"type": "string"}, "probability": {"type": "number"}}}},
            "selected_symptoms": {"type": "array", "items": {"type": "string"}},
            "explanation": {"type": "array", "description": "Change in the predicted probability when each symptom is removed.",
                            "items": {"type": "object"}},
            "low_confidence": {"type": "boolean"},
            "model_version": {"type": "string"},
            "trained_on_synthetic_data": {"type": "boolean"},
            "disclaimer": {"type": "string"},
            "id": {"type": "integer", "description": "History id (only when saved)."},
            "result_url": {"type": "string", "description": "Result page URL (only when saved)."}}},
        "Error": {"type": "object", "properties": {"error": {"type": "string"}}},
    }},
}
