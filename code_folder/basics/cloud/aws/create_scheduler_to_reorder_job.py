"""
this is the code to reschedule/reorder kubernetes job based on the tag/label on the job
"""

from kubernetes import client, config

# Load Kubernetes configuration
config.load_kube_config()

# Create a Kubernetes API client
api_client = client.BatchV1Api()

# Define the label selector for filtering jobs
label_selector = {
    "label-key1": "value1",
    "label-key2": "value2",
    "label-key3": "value3",
}

# Get the list of jobs matching the specified labels
jobs = api_client.list_namespaced_job(
    namespace="your-namespace", label_selector=label_selector
)

# Iterate over each job and perform the reorganization logic
for job in jobs.items:
    # Perform your reorganization logic for each job here
    # For example, you can update the labels or move the job to a different namespace
    # Replace the placeholders with the actual logic specific to your use case
    print(f"Reorganizing job: {job.metadata.name}")
    # Your reorganization logic goes here
