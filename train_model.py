import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from preprocess_utils import preprocess_df

MODEL_PATH = "model.pkl"

def train_and_save():
    df = pd.read_csv("data/dataset.csv")
    df = preprocess_df(df)

    target = "Disease_Prediction"
    if target not in df.columns:
        raise ValueError("Disease_Prediction column not found in dataset")

    X = df.drop(columns=[target])
    y = df[target]

    numeric_features = [
        "Age",
        "Weight",
        "Duration_Days",
        "Body_Temperature",
        "Heart_Rate",
    ]
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(
        steps = [
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestClassifier(
        random_state=42
    )

    clf = Pipeline(
        steps=[
            ("prep", FunctionTransformer(preprocess_df, validate=False)),
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    class_counts = y.value_counts()
    if class_counts.min() < 2:
        # Too few samples in some classes for stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    payload = {
        "model": clf,
        "feature_columns": list(X.columns),
    }
    joblib.dump(payload, MODEL_PATH)


if __name__ == "__main__":
    train_and_save()
    print(f"Model saved to {MODEL_PATH}")
