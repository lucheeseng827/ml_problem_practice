import datadog

# Set the Datadog API key and the application key
datadog.api_key = "your-api-key"
datadog.app_key = "your-application-key"

# Send a metric
datadog.statsd.gauge("metric_name", value, tags=["tag1", "tag2"])
