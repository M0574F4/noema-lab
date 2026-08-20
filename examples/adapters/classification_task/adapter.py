from __future__ import annotations


def load_classification_examples(request):
    return {
        "metadata": {"dataset": request["params"].get("dataset", "example_external_classification")},
        "examples": [
            {"id": "clear_001", "label": "clear", "prediction": "clear"},
            {"id": "fade_001", "label": "faded", "prediction": "clear"},
            {"id": "block_001", "label": "blocked", "prediction": "blocked"},
            {"id": "noise_001", "label": "noisy", "prediction": "noisy"},
        ],
    }


def score_classification(request):
    reference = {item["id"]: item.get("label", "") for item in request["reference"]}
    candidate = {item["id"]: item.get("prediction", item.get("label", "")) for item in request["candidate"]}
    rows = []
    correct = 0
    for example_id, expected in reference.items():
        predicted = candidate.get(example_id, "")
        match = str(expected).strip().lower() == str(predicted).strip().lower()
        correct += 1 if match else 0
        rows.append(
            {
                "id": example_id,
                "expected": expected,
                "predicted": predicted,
                "exact_match": 1.0 if match else 0.0,
            }
        )
    total = len(rows) or 1
    accuracy = float(correct) / float(total)
    return {
        "metric_family": "external_classification",
        "metrics": {
            "external.classification.accuracy": accuracy,
            "task.accuracy": accuracy,
        },
        "per_example": rows,
    }
