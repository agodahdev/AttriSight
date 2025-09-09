from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def make_preprocessor(num_features, cat_features):
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_features),
        ("cat", cat_pipe, cat_features)
    ])

def make_logreg_pipeline(num_features, cat_features):
    return Pipeline([
        ("pre", make_preprocessor(num_features, cat_features)),
        ("clf", LogisticRegression(max_iter=1000))
    ])

def make_rf_pipeline(num_features, cat_features):
    return Pipeline([
        ("pre", make_preprocessor(num_features, cat_features)),
        ("clf", RandomForestClassifier(random_state=42))
    ])