# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from datetime import datetime

predictor_model = joblib.load('obstacle_predictor.pkl')
print("Obstacle prediction model loaded.")

# 1. Create fake data
print("Generating fake training data...")
data = []
for _ in range(10000):
    hour = np.random.randint(0, 24)
    day_of_week = np.random.randint(0, 7) # 0=Monday, 6=Sunday
    is_rush_hour = 1 if (hour >= 7 and hour <= 9) or (hour >= 17 and hour <= 19) else 0
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # Let's assume obstacles are more likely during rush hour on weekdays
    obstacle_prob = 0.1
    if is_rush_hour and not is_weekend:
        obstacle_prob = 0.6
    
    has_obstacle = 1 if np.random.rand() < obstacle_prob else 0
    data.append([hour, day_of_week, is_rush_hour, is_weekend, has_obstacle])

df = pd.DataFrame(data, columns=['hour', 'day_of_week', 'is_rush_hour', 'is_weekend', 'has_obstacle'])

# 2. Train the model
print("Training obstacle prediction model...")
X = df[['hour', 'day_of_week', 'is_rush_hour', 'is_weekend']]
y = df['has_obstacle']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"Model Accuracy: {model.score(X_test, y_test):.2f}")

# 3. Save the trained model
joblib.dump(model, 'obstacle_predictor.pkl')
print("Model saved as obstacle_predictor.pkl")