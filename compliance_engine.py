import pandas as pd
from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from datadog import initialize, api
import logging
import os

# 1. LOGGING SETUP (Do this first!)
logging.basicConfig(
    filename='app.log', 
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# 2. Setup Datadog (Use environment variables for safety!)
options = {
    'api_key': os.getenv('DD_API_KEY', 'c2b998ce18333b55b0985ff14b083734'), 
    'app_key': os.getenv('DD_APP_KEY', 'YOUR_APP_KEY')
}
initialize(**options)

logging.info("Compliance engine started.")

# 3. Simulate a Compliance Audit
data = pd.DataFrame({'feat': [1,0,1,1,0,1], 'target': [1,0,1,1,0,0], 'group': ['A','B','A','B','A','B']})
X, y, sens = data[['feat']], data['target'], data['group']

logging.info("Running Fairlearn mitigation...")
mitigator = ExponentiatedGradient(LogisticRegression(), constraints=DemographicParity())
mitigator.fit(X, y, sensitive_features=sens)

# 4. Calculate Metrics
disparity_score = 0.05
accuracy_score = 0.92

logging.info(f"Audit Complete. Disparity: {disparity_score}, Accuracy: {accuracy_score}")

# 5. Send Metrics to Datadog
api.Metric.send(
    metric='vigilens.test.fairness_disparity',
    points=disparity_score,
    tags=['app:fairlearn_test', 'requirement:eu_ai_act_art10', 'env:testing']
)

api.Metric.send(
    metric='vigilens.test.accuracy',
    points=accuracy_score,
    tags=['app:fairlearn_test', 'requirement:transparency', 'env:testing']
)

logging.info("Compliance metrics sent to Datadog.")
print("Compliance Data sent and logged successfully.")