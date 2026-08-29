import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# File paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "slip_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "sensor_data.csv")


# --------------------------------------------------
# Load model and dataset
# --------------------------------------------------

model = joblib.load(MODEL_PATH)
data = pd.read_csv(DATA_PATH)

print("Model loaded successfully.")
print("Dataset loaded successfully.")
print("Number of samples:", len(data))


# --------------------------------------------------
# Create derived features
# --------------------------------------------------

data["speed_difference"] = (
    data["wheel_speed"] - data["imu_speed"]
).abs()

data["accel_magnitude"] = (
    data["accel_x"] ** 2 +
    data["accel_y"] ** 2
) ** 0.5


# --------------------------------------------------
# Select features
# --------------------------------------------------

features = [
    "accel_x",
    "accel_y",
    "gyro_z",
    "imu_variance",
    "wheel_speed",
    "imu_speed",
    "speed_difference",
    "accel_magnitude"
]

X = data[features]
y = data["slip"]


# --------------------------------------------------
# Split dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nDataset Split")
print("------------------------------")
print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# --------------------------------------------------
# IMPORTANT
# --------------------------------------------------
# The existing model was already trained on the
# complete dataset.
#
# Therefore, this script cannot honestly use the
# existing model to evaluate the unseen test set.
#
# We train a fresh model using ONLY the training set.
# --------------------------------------------------

from sklearn.ensemble import RandomForestClassifier

test_model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    n_jobs=-1
)

test_model.fit(X_train, y_train)


# --------------------------------------------------
# Predict on unseen test data
# --------------------------------------------------

predictions = test_model.predict(X_test)


# --------------------------------------------------
# Accuracy
# --------------------------------------------------

accuracy = accuracy_score(y_test, predictions)

print("\n------------------------------")
print("UNSEEN TEST PERFORMANCE")
print("------------------------------")

print(
    "Test Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# --------------------------------------------------
# Classification report
# --------------------------------------------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# --------------------------------------------------
# Sample predictions
# --------------------------------------------------

print("\nSample Test Predictions:")
print("------------------------------")

for i in range(min(10, len(X_test))):

    actual = y_test.iloc[i]
    predicted = predictions[i]

    print(
        f"Sample {i + 1}: "
        f"Actual = {actual}, "
        f"Predicted = {predicted}"
    )