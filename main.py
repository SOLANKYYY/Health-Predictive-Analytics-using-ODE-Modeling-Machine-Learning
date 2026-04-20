import numpy as np
import pandas as pd
from scipy.integrate import odeint
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Define the ODE model for heart rate dynamics
def heart_rate_dynamics(y, t, parameters):
    # parameters[0] = baseline heart rate, parameters[1] = rate of increase due to activity, parameters[2] = rate of decrease due to rest
    baseline, activity_factor, rest_factor = parameters
    heart_rate = y[0]
    
    # Assuming a simple model where activity increases HR and rest decreases HR
    if activity_factor > 0:
        dHR_dt = activity_factor * (activity_factor - heart_rate)  # Increase HR during activity
    else:
        dHR_dt = rest_factor * (baseline - heart_rate)  # Decrease HR during rest
        
    return [dHR_dt]

# Generate heart rate data based on ODE model
def generate_heart_rate_data(baseline, activity_factor, rest_factor, time_points):
    initial_conditions = [baseline]  # Initial heart rate
    parameters = [baseline, activity_factor, rest_factor]  # ODE parameters
    result = odeint(heart_rate_dynamics, initial_conditions, time_points, args=(parameters,))
    return result[:, 0]

# Simulating other factors like age, BMI, physical activity, etc.
def get_user_input():
    print("Enter your details to predict potential diseases:")
    age = float(input("Age (years): "))
    bmi = float(input("BMI (kg/m²): "))
    heart_rate = float(input("Current heart rate (bpm): "))
    activity_level = float(input("Activity level (0 = rest, 1 = moderate activity, 2 = intense activity): "))
    sleep_hours = float(input("Sleep hours per night: "))
    blood_pressure = float(input("Blood pressure (systolic/diastolic): "))
    
    return [age, bmi, heart_rate, activity_level, sleep_hours, blood_pressure]

# Machine learning model for disease prediction
def disease_prediction_model(X_train, y_train):
    scaler = StandardScaler()  # Standardize the features
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    return model, scaler

# Predict disease based on user input
def predict_disease(model, scaler, user_input):
    user_input_scaled = scaler.transform([user_input])
    prediction = model.predict(user_input_scaled)
    diseases = ['No Disease', 'Hypertension', 'Diabetes', 'Heart Disease', 'Sleep Apnea']
    print(f"Predicted Disease: {diseases[int(prediction[0])]}")
    
# Generate simulated data for training
def generate_simulated_data():
    # Simulating some training data
    data = []
    labels = []
    
    for _ in range(1000):
        age = np.random.randint(18, 80)
        bmi = np.random.uniform(18.5, 40)
        heart_rate = np.random.uniform(60, 100)
        activity_level = np.random.choice([0, 1, 2])
        sleep_hours = np.random.uniform(4, 10)
        blood_pressure = np.random.uniform(90, 180)
        
        # Using ODE model to simulate heart rate dynamics for a period
        time_points = np.linspace(0, 10, 100)
        hr_data = generate_heart_rate_data(heart_rate, activity_level, -0.1, time_points)
        
        # Simulate disease labels based on certain conditions
        if heart_rate > 90 and bmi > 30:
            label = 2  # Possible heart disease
        elif blood_pressure > 140:
            label = 1  # Hypertension
        elif bmi > 30 and sleep_hours < 6:
            label = 3  # Sleep apnea
        else:
            label = 0  # No disease
        
        data.append([age, bmi, heart_rate, activity_level, sleep_hours, blood_pressure])
        labels.append(label)
    
    return np.array(data), np.array(labels)

# Main program
if __name__ == "__main__":
    # Generate simulated data
    X, y = generate_simulated_data()
    
    # Split data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train machine learning model
    model, scaler = disease_prediction_model(X_train, y_train)
    
    # Get user input and make prediction
    user_input = get_user_input()
    predict_disease(model, scaler, user_input)
