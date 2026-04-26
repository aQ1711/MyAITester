import pandas as pd
from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

# 1. Create dummy data so the script has something to train on
# In your real system, you would load this from a CSV or Database
data = pd.DataFrame({
    'feature1': [1, 0, 1, 1, 0, 0, 1, 0],
    'feature2': [0, 1, 0, 0, 1, 1, 0, 1],
    'target':   [1, 0, 1, 1, 0, 0, 1, 0],
    'gender':   ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female']
})

X_train = data[['feature1', 'feature2']]
y_train = data['target']
sensitive_features = data['gender']

# 2. Define your base model
base_model = LogisticRegression(solver='liblinear')

# 3. Define the Compliance Rule
constraint = DemographicParity()

# 4. The "Fairness-Aware" Trainer
mitigator = ExponentiatedGradient(base_model, constraints=constraint)

# 5. Train the model
mitigator.fit(X_train, y_train, sensitive_features=sensitive_features)

print("Compliance training complete. Model is now constraint-aware.")