import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# LOAD DATASET
# ============================================================

data = pd.read_csv("sensor_data.csv")

print("Dataset loaded successfully!")
print("Shape:", data.shape)


# ============================================================
# FEATURES AND TARGET
# ============================================================

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


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# TRAIN RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("\nModel training completed!")


# ============================================================
# MODEL PREDICTION
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Slip", "Slip"]
)

disp.plot()
plt.title("Wheel Slip Detection - Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(importance)

plt.figure(figsize=(8, 5))
plt.barh(importance["Feature"], importance["Importance"])
plt.xlabel("Importance")
plt.title("Feature Importance for Slip Detection")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(model, "slip_model.pkl")

print("\nModel saved successfully as slip_model.pkl")