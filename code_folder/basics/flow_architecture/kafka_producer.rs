use rdkafka::config::ClientConfig;
use rdkafka::producer::{FutureProducer, FutureRecord};

#[tokio::main]
async fn main() {
    let producer: FutureProducer = ClientConfig::new()
        .set("bootstrap.servers", "localhost:9092") // replace with your Kafka broker address
        .create()
        .expect("Producer creation error");

    let message_to_send = "Your message with metadata";

    let result = producer
        .send(
            FutureRecord::to("topic-name") // replace with your topic name
                .payload(message_to_send)
                .key("message-key"), // replace with your key
            5000,
        )
        .await;

    match result {
        Ok(delivery) => println!("Message delivered to {:?}", delivery),
        Err((err, _)) => println!("Error producing message: {:?}", err),
    }
}
