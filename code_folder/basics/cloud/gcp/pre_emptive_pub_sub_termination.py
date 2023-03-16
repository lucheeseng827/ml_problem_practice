# Create a Pub/Sub topic that will be used to send messages to stop the preemptible VMs.
# Deploy a Python script that listens to the Pub/Sub topic and stops the preemptible VMs when a message is received.
# Send a message to the Pub/Sub topic to trigger the preemptible VM termination.

""""
gcloud components install
gcloud services enable compute.googleapis.com
gcloud services enable pubsub.googleapis.com

gcloud pubsub topics create stop-preemptible-vm

pip install google-cloud-pubsub google-cloud-compute

"""


import time
import os
from google.cloud import pubsub_v1, compute_v1

# Function to stop a preemptible VM


def stop_preemptible_vm(project, zone, instance):
    compute_client = compute_v1.InstancesClient()
    compute_client.stop(project=project, zone=zone, instance=instance)

# Callback function to process the received message


def callback(message):
    project = message.attributes['project']
    zone = message.attributes['zone']
    instance = message.attributes['instance']
    stop_preemptible_vm(project, zone, instance)
    message.ack()


# Parameters
project_id = 'your-project-id'
subscription_id = 'stop-preemptible-vm-subscription'

# Create a Pub/Sub subscriber
subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(project_id, subscription_id)

# Create the subscription if it does not exist
try:
    subscriber.get_subscription(request={"subscription": subscription_path})
except Exception:
    topic_path = subscriber.topic_path(project_id, 'stop-preemptible-vm')
    subscriber.create_subscription(
        request={"name": subscription_path, "topic": topic_path})

# Listen to the Pub/Sub topic
subscriber.subscribe(subscription_path, callback=callback)

# Keep the script running
print("Listening for messages to stop preemptible VMs...")
while True:
    time.sleep(60)


"""
gcloud pubsub topics publish stop-preemptible-vm --message "Stop VM" \
    --attribute project=your-project-id,zone=your-zone,instance=your-instance-name

"""
