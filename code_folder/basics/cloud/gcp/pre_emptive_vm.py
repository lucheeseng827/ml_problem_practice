#how to get prememptive vm in gcp to start and stop ahead of schedule of termination

import time
from google.cloud import compute_v1

# Function to fetch regular VM pricing
def get_regular_vm_pricing(client, project, region, instance_type):
    sku_list = client.list_skus(project=project, service='compute.googleapis.com', parent=f"projects/{project}/services/compute.googleapis.com")
    for sku in sku_list:
        if sku.category.resource_group == 'GCEInstance' and sku.category.usage_type == 'OnDemand':
            if instance_type in sku.description:
                return float(sku.pricing_info[0].pricing_expression.tiered_rates[0].unit_price.nanos) / 1e9

    return None

# Parameters
project_id = 'your-project-id'
region = 'us-central1'
instance_type = 'n1-standard-1'
preemptible_discount_rate = 0.7  # Discount rate for preemptible VMs
monitor_interval = 60  # seconds

# Create a Compute Engine client
client = compute_v1.CloudBillingClient()

# Monitor preemptible VM pricing
while True:
    regular_price = get_regular_vm_pricing(client, project_id, region, instance_type)

    if regular_price is not None:
        preemptible_price = regular_price * (1 - preemptible_discount_rate)
        print(f"Preemptible VM Price for {instance_type} in {region}: ${preemptible_price:.4f}")

    time.sleep(monitor_interval)
