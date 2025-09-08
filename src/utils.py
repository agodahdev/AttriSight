import json

def yes_no_to_binary(series):
    return series.map({"Yes": 1, "No": 0})

def save_feature_order(path, columns):
    path.write_text(json.dumps(columns))

def load_feature_order(path):
    return json.loads(path.read_text())