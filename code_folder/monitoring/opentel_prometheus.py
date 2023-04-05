from prometheus_client import Summary, push_to_gateway
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PrometheusExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

# Initialize the OpenTelemetry components
metrics.set_meter_provider(MeterProvider())
trace.set_tracer_provider(TracerProvider())

# Create a Summary metric to track the average response time
response_time = Summary("response_time", "Average response time")

# Record the response time for a specific request


@response_time.time()
def handle_request():
    # Perform some work and return the response
    return response


# Push the collected metrics to the Prometheus gateway
exporter = PrometheusExporter(namespace="my_app")
push_to_gateway("prometheus-server:9090", job="my-app", registry=exporter.registry)
