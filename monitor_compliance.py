import logging
from datadog import initialize, statsd
from fairlearn.metrics import demographic_parity_difference

# 1. Setup Logging (matches Datadog's configuration)
logging.basicConfig(filename='/var/log/datadog_custom/app.log', level=logging.INFO)

# 2. Initialize Datadog StatsD (Local agent runs on localhost:8125)
initialize(statsd_host='127.0.0.1', statsd_port=8125)

def check_ai_compliance(y_true, y_pred, sensitive_features):
    # Calculate fairness metric
    bias = demographic_parity_difference(
        y_true, y_pred, sensitive_features=sensitive_features
    )
    
    # Send metric to Datadog dashboard
    statsd.gauge('fairlearn.bias.parity', bias)
    
    # Log audit trail for EU AI Act Article 12 (Transparency/Record keeping)
    if bias > 0.05: # Threshold of 5% bias
        msg = f"NON_COMPLIANT: High bias detected at {bias:.4f}"
        logging.error(msg)
    else:
        msg = f"COMPLIANT: Bias level at {bias:.4f}"
        logging.info(msg)
    
    print(f"Compliance check complete: {msg}")

if __name__ == "__main__":
    # Mock data for demonstration
    y_t = [0, 1, 1, 0, 1, 0]
    y_p = [0, 0, 1, 0, 1, 1]
    s_f = [0, 1, 0, 1, 0, 1]
    
    check_ai_compliance(y_t, y_p, s_f)
