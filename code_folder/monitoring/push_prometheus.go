package main

import (
  "fmt"
  "net/http"

  "github.com/prometheus/client_golang/prometheus"
  "github.com/prometheus/client_golang/prometheus/promauto"
  "github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
  // Create a Histogram metric to track the request latencies
  requestLatency = promauto.NewHistogram(prometheus.HistogramOpts{
    Name: "request_latency",
    Help: "Request latencies",
  })
)

func main() {
  // Register the metrics with Prometheus
  prometheus.MustRegister(requestLatency)

  // Collect some data for the metric
  requestLatency.Observe(0.3)
  requestLatency.Observe(0.5)

  // Start the HTTP server to expose the metrics
  http.Handle("/metrics", promhttp.Handler())
  fmt.Println("Starting server on port 8080")
  http.ListenAndServe(":8080", nil)
}

// The above code will expose the metrics at http://localhost:8080/metrics. You can see the metrics in the browser or using curl:
