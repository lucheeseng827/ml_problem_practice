package main

import (
	"fmt"

	"github.com/Shopify/sarama"
)

func main() {
	// Define the Kafka broker addresses
	brokers := []string{"localhost:9092"}

	// Create a Kafka producer configuration
	config := sarama.NewConfig()
	config.Producer.RequiredAcks = sarama.WaitForLocal       // Wait for acknowledgement from the Kafka broker
	config.Producer.Compression = sarama.CompressionSnappy   // Use Snappy compression
	config.Producer.Flush.Frequency = 500 * time.Millisecond // Flush messages every 500ms

	// Create a Kafka producer
	producer, err := sarama.NewAsyncProducer(brokers, config)
	if err != nil {
		panic(err)
	}
	defer producer.Close()

	// Create a Kafka message
	message := &sarama.ProducerMessage{
		Topic: "my_topic",
		Value: sarama.StringEncoder("Hello, Kafka!"),
	}

	// Send the message to Kafka
	producer.Input() <- message

	// Wait for the message to be sent (or for an error to occur)
	select {
	case err := <-producer.Errors():
		panic(err)
	case <-producer.Successes():
		fmt.Println("Message sent to Kafka!")
	}
}
