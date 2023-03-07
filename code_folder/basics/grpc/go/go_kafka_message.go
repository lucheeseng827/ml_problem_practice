syntax = "proto3";

package kafka;

service KafkaService {
  rpc Produce (KafkaMessage) returns (google.protobuf.Empty) {}
}

message KafkaMessage {
  string topic = 1;
  bytes message = 2;
}



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
