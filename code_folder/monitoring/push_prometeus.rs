use prometheus::{Histogram, Opts, Registry, TextEncoder, Encoder};

// Create a Histogram metric to track the request latencies
let histogram = Histogram::with_opts(Opts::new("request_latency", "Request latencies")).unwrap();

// Register the metric with a registry
let registry = Registry::new();
registry.register(Box::new(histogram.clone())).unwrap();

// Collect some data for the metric
histogram.observe(0.3);
histogram.observe(0.5);

// Encode the metrics in Prometheus' text format
let mut buffer = vec![];
let encoder = TextEncoder::new();
let metric_families = registry.gather();
encoder.encode(&metric_families, &mut buffer).unwrap();

// Print the encoded metrics
println!("{}", String::from_utf8(buffer).
