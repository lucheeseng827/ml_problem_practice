"""To acknowledge or resolve an incident, you can use the /incidents/{id}/acknowledge and /incidents/{id}/resolve endpoints of the PagerDuty Events API, respectively.

For more information about the PagerDuty Events API and other ways to send notifications to PagerDuty using Python, you can refer to the documentation: https://v2.developer.pagerduty.com/docs/events-api-v2"""

import requests

# Set the PagerDuty API token and the service key
PAGERDUTY_TOKEN = "your-api-token"
SERVICE_KEY = "your-service-key"

# Set the incident details
incident = {"summary": "Error in Production", "severity": "critical", "source": "api"}

# Send the incident
response = requests.post(
    "https://api.pagerduty.com/incidents",
    headers={
        "Authorization": f"Token token={PAGERDUTY_TOKEN}",
        "Content-Type": "application/json",
    },
    json={"service_key": SERVICE_KEY, "incident": incident},
)

# Check the status code of the response
if response.status_code == 201:
    print("Incident triggered successfully.")
else:
    print("Error triggering incident:", response.status_code)
