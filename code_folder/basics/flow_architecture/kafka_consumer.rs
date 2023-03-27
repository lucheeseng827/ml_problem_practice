use rdkafka::config::{ClientConfig, RDKafkaLogLevel};
use rdkafka::consumer::stream_consumer::StreamConsumer;
use rdkafka::consumer::{CommitMode, Consumer};
use rdkafka::error::KafkaResult;
use rdkafka::message::{Headers, Message};

fn main() -> KafkaResult<()> {
    // Create a consumer configuration
    let consumer_config = ClientConfig::new()
        .set("bootstrap.servers", "localhost:9092")
        .set("group.id", "my_consumer_group")
        .set("enable.partition.eof", "false")
        .set("session.timeout.ms", "6000")
        .set("enable.auto.commit", "true")
        .set("auto.commit.interval.ms", "100")
        .set("log.connection.close", "false")
        .set("log.level", RDKafkaLogLevel::Debug)
        .create();

    // Create a consumer
    let mut consumer: StreamConsumer = consumer_config.create()?;

    // Subscribe to a topic
    consumer.subscribe(&["my_topic"])?;

    // Poll for messages indefinitely
    loop {
        match consumer.poll(Duration::from_millis(100)) {
            Ok(None) => continue,
            Ok(Some(message)) => {
                // Process the message
                match message.payload_view::<str>() {
                    Some(Ok(s)) => println!(
                        "Key: '{}', Value: '{}'",
                        message.key_view::<str>().unwrap(),
                        s
                    ),
                    Some(Err(e)) => eprintln!("Error while deserializing message payload: {}", e),
                    None => eprintln!("Message payload is not a string"),
                }
            }
            Err(e) => eprintln!("Error while consuming message: {}", e),
        }
    }
}
