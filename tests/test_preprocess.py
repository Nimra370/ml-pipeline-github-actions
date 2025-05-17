from src import preprocess

def test_data_shape():
    df = preprocess.load_data()
    assert df.shape[1] == 5  # 4 features + 1 label

def test_split():
    df = preprocess.load_data()
    X_train, X_test, y_train, y_test = preprocess.preprocess_data(df)
    assert len(X_train) > 0 and len(X_test) > 0
