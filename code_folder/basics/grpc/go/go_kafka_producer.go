syntax = "proto3";

package sarama;

type kafkaServer struct {
  producer *kafka.Producer
}


func (s *kafkaServer) Produce(ctx context.Context, req *kafka.KafkaMessage) (*google_protobuf.Empty, error) {
    msg := &kafka.Message{
        Topic: req.Topic,
        Value: req.Message,
    }
    _, _, err := s.producer.SendMessage(msg)
    if err != nil {
        return nil, err
    }
    return &google_protobuf.Empty{}, nil
}
