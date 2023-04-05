package (
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/push"
)

// Create a Histogram metric to track the request latencies
histogram := prometheus.NewHistogram(prometheus.HistogramOpts{
    Name: "request_latency",
    Help: "Request latencies",
})

// Register the metric with the default registry
prometheus.MustRegister(histogram)

// Collect some data for the metric
histogram.Observe(0.3)
histogram.Observe(0.5)

// Push the collected metrics to the Prometheus gateway
err := push.New("http://prometheus-server:9090", "my-app").Push()
if err != nil {
    panic(err)
}
