import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

heart_data = pd.read_csv("heart_disease_data.csv")

print("\nFirst 5 Rows:")
print(heart_data.head())

print("\nDataset Shape:")
print(heart_data.shape)

print("\nTarget Count:")
print(heart_data['target'].value_counts())

X = heart_data.drop(columns='target')
Y = heart_data['target']

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled,
    Y,
    test_size=0.2,
    random_state=2,
    stratify=Y
)

model = LogisticRegression()

model.fit(X_train, Y_train)

train_prediction = model.predict(X_train)
train_accuracy = accuracy_score(Y_train, train_prediction)

test_prediction = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, test_prediction)

print("\nTraining Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

while True:

    print("\n===== Heart Disease Prediction =====")

    age = float(input("Enter Age: "))
    sex = float(input("Enter Sex (1=Male, 0=Female): "))
    cp = float(input("Enter Chest Pain Type (0-3): "))
    trestbps = float(input("Enter Resting Blood Pressure: "))
    chol = float(input("Enter Cholesterol: "))
    fbs = float(input("Enter Fasting Blood Sugar (1=True, 0=False): "))
    restecg = float(input("Enter Rest ECG Result (0-2): "))
    thalach = float(input("Enter Max Heart Rate Achieved: "))
    exang = float(input("Exercise Induced Angina (1=Yes, 0=No): "))
    oldpeak = float(input("Enter ST Depression: "))
    slope = float(input("Enter Slope (0-2): "))
    ca = float(input("Enter Number of Major Vessels (0-4): "))
    thal = float(input("Enter Thalassemia (0-3): "))

    input_data = (
        age, sex, cp, trestbps, chol, fbs,
        restecg, thalach, exang, oldpeak,
        slope, ca, thal
    )

    input_df = pd.DataFrame([input_data], columns=X.columns)

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)

    print("\nPrediction Result:")

    if prediction[0] == 0:
        print("No Heart Disease")
    else:
        print("Heart Disease Detected")

    again = input("\nCheck another person? (yes/no): ")

    if again.lower() != "yes":
        print("Program Ended.")
        break