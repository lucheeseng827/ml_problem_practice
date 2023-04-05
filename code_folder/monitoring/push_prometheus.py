from urllib import response
from prometheus_client import Summary, push_to_gateway

# Create a Summary metric to track the average response time
response_time = Summary("response_time", "Average response time")


# Record the response time for a specific request
@response_time.time()
def handle_request():
    # Perform some work and return the response
    return response


# Push the collected metrics to the Prometheus gateway
push_to_gateway("prometheus-server:9090", job="my-app")
