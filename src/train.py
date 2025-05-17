import joblib
from sklearn.ensemble import RandomForestClassifier
from src.preprocess import load_data, preprocess_data

def train_model():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Accuracy: {score}")
    joblib.dump(model, "model/model.pkl")
    return score

if __name__ == "__main__":
    train_model()
