import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset (with low_memory=False to fix dtype warnings)
df = pd.read_csv("data/airline_1.csv", low_memory=False)

# Drop 'Unnamed: 0' if present
df = df.drop("Unnamed: 0", axis=1, errors="ignore")

# Label encode all object-type columns
le = LabelEncoder()
for i in df.columns:
    if df[i].dtypes == object:
        df[i] = le.fit_transform(df[i].astype(str))

# Define features and target (using correct casing for 'Satisfaction')
x = df.drop(["Satisfaction"], axis=1)
y = df["Satisfaction"]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=1)

# Train Random Forest model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict and print accuracy
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# Save the model to file
joblib.dump(model, "model.joblib")


