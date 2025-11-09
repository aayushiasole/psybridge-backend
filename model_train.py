import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
import joblib
import os

# --- Load dataset ---
DATASET_PATH = "data/mbti_dataset.csv"
df = pd.read_csv(DATASET_PATH, encoding="ISO-8859-1")

# --- Prepare data ---
question_cols = [col for col in df.columns if col not in ['Response Id', 'Personality']]
X = df[question_cols].values
y = df['Personality'].values

# --- Encode target ---
le = LabelEncoder()
y_enc = le.fit_transform(y)

# --- Split and train ---
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)
clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=42)
clf.fit(X_train, y_train)

# --- Save model ---
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/mlp_model.joblib")
joblib.dump(le, "models/mbti_labelencoder.joblib")

print("✅ Model and label encoder saved successfully!")
