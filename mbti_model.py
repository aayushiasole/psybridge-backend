import joblib

def load_model():
    clf = joblib.load("models/mlp_model.joblib")
    le = joblib.load("models/mbti_labelencoder.joblib")
    return clf, le

def predict_mbti(answers):
    clf, le = load_model()
    pred = clf.predict([answers])
    mbti_type = le.inverse_transform(pred)[0]
    return mbti_type
