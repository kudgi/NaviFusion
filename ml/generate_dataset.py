import numpy as np
import pandas as pd

# Reproducibility
np.random.seed(42)

N_NORMAL = 5000
N_SLIP = 5000


# ============================================================
# NORMAL DRIVING DATA
# ============================================================

normal_wheel_speed = np.random.uniform(0.5, 5.0, N_NORMAL)

# IMU speed is close to wheel speed during normal driving
normal_imu_speed = normal_wheel_speed + np.random.normal(
    0, 0.15, N_NORMAL
)

normal_accel_x = np.random.normal(0, 0.4, N_NORMAL)
normal_accel_y = np.random.normal(0, 0.25, N_NORMAL)

normal_gyro_z = np.random.normal(0, 0.2, N_NORMAL)

# Stable IMU readings
normal_imu_variance = np.random.uniform(0.01, 0.15, N_NORMAL)


# ============================================================
# WHEEL SLIP DATA
# ============================================================

slip_wheel_speed = np.random.uniform(0.5, 5.0, N_SLIP)

# During slip, wheel encoder speed differs significantly
slip_imu_speed = slip_wheel_speed + np.random.normal(
    0, 1.0, N_SLIP
)

slip_accel_x = np.random.normal(0, 1.2, N_SLIP)
slip_accel_y = np.random.normal(0, 0.8, N_SLIP)

slip_gyro_z = np.random.normal(0, 0.7, N_SLIP)

# Higher IMU instability
slip_imu_variance = np.random.uniform(0.15, 1.0, N_SLIP)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

normal_speed_difference = np.abs(
    normal_wheel_speed - normal_imu_speed
)

slip_speed_difference = np.abs(
    slip_wheel_speed - slip_imu_speed
)


normal_accel_magnitude = np.sqrt(
    normal_accel_x**2 + normal_accel_y**2
)

slip_accel_magnitude = np.sqrt(
    slip_accel_x**2 + slip_accel_y**2
)


# ============================================================
# CREATE DATAFRAMES
# ============================================================

normal_data = pd.DataFrame({
    "accel_x": normal_accel_x,
    "accel_y": normal_accel_y,
    "gyro_z": normal_gyro_z,
    "imu_variance": normal_imu_variance,
    "wheel_speed": normal_wheel_speed,
    "imu_speed": normal_imu_speed,
    "speed_difference": normal_speed_difference,
    "accel_magnitude": normal_accel_magnitude,
    "slip": 0
})


slip_data = pd.DataFrame({
    "accel_x": slip_accel_x,
    "accel_y": slip_accel_y,
    "gyro_z": slip_gyro_z,
    "imu_variance": slip_imu_variance,
    "wheel_speed": slip_wheel_speed,
    "imu_speed": slip_imu_speed,
    "speed_difference": slip_speed_difference,
    "accel_magnitude": slip_accel_magnitude,
    "slip": 1
})


# ============================================================
# COMBINE AND SHUFFLE
# ============================================================

data = pd.concat([normal_data, slip_data])

data = data.sample(frac=1, random_state=42).reset_index(drop=True)


# ============================================================
# SAVE DATASET
# ============================================================

data.to_csv("sensor_data.csv", index=False)

print("Dataset generated successfully!")
print("Total samples:", len(data))
print("\nClass distribution:")
print(data["slip"].value_counts())

print("\nFirst 5 rows:")
print(data.head())