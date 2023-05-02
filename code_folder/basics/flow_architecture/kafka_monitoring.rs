use opentelemetry::{KeyValue, global, sdk::propagation::TraceContextPropagator};
use opentelemetry_contrib::kafka::{
    KafkaEvent,
    KafkaProducerInstrument,
    KafkaSubscriberInstrument,
};
use rdkafka::{
    ClientConfig,
    config::ClientConfigExt,
    message::ToBytes,
    producer::{FutureProducer, FutureRecord},
    subscriber::{Consumer, ConsumerContext, Rebalance, Subscription},
};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize OpenTelemetry tracer
    let tracer = global::tracer("my-kafka-streams");
    let propagator = TraceContextPropagator::new();
    global::set_text_map_propagator(propagator);

    // Create Kafka producer
    let producer: FutureProducer = ClientConfig::new()
        .set("bootstrap.servers", "localhost:9092")
        .set("message.timeout.ms", "5000")
        .create()
        .expect("failed to create producer");

    // Create Kafka subscriber
    let consumer: Consumer<KafkaSubscriberInstrument> = ClientConfig::new()
        .set("group.id", "my-consumer-group")
        .set("bootstrap.servers", "localhost:9092")
        .set_default_topic_config(
            ClientConfig::default()
                .set("auto.offset.reset", "earliest")
                .get("default_topic_config")
                .expect("failed to get default_topic_config")
                .clone()
        )
        .create_with_context(KafkaSubscriberInstrument)
        .expect("failed to create consumer");

    // Subscribe to Kafka topic
    consumer.subscribe(&[Subscription::topic("my-topic")])?;

    // Wait for Kafka messages and trace them
    while let Some(message) = consumer.poll(Duration::from_millis(100)).unwrap() {
        if let Some(payload) = message.payload() {
            // Extract tracing context from message headers
            let context = propagator.extract(&message.headers());

            // Start new span
            let span = tracer
                .start("process_message", Some(context), vec![
                    KeyValue::new("kafka.topic", message.topic()),
                    KeyValue::new("kafka.partition", message.partition().to_string()),
                    KeyValue::new("kafka.offset", message.offset().to_string()),
                ]);

            // Process message
            println!("Received message: {:?}", String::from_utf8_lossy(payload));

            // Propagate tracing context to Kafka producer
            let mut headers = rdkafka::producer::BaseHeaders::default();
            propagator.inject_context(&span.context(), &mut headers);
            let record = FutureRecord::to("my-topic")
                .headers(headers)
                .key(message.key().unwrap_or_default().as_ref())
                .payload(payload);

            // Produce message
            producer.send(record, 0).await?;

            // End span
            span.end();
        }
    }

    Ok(())
}
