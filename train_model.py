# train_model.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# Sample synthetic dataset
data = {
    "income": [3000000, 7000000, 2500000, 8000000, 4500000],
    "age": [25, 40, 22, 50, 35],
    "creditworthy": [0, 1, 0, 1, 1]
}
df = pd.DataFrame(data)

X = df[["income", "age"]]
y = df["creditworthy"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model/model.pkl")
