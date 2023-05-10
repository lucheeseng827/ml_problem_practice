package main

import (
	"log"
	"net"
	"google.golang.org/grpc"
	"kafka/package"
)


type kafkaServer struct {
	kafka.UnimplementedKafkaServiceServer             // Embed the unimplemented methods so the server won't break when methods are added but not implemented
	producer                              interface{} // Replace with the actual type of your producer
}

// Implement the required methods from the KafkaServiceServer interface
// For example:
// func (s *kafkaServer) SomeRPCMethod(ctx context.Context, req *kafka.SomeRequest) (*kafka.SomeResponse, error) {
// 	 // Your method implementation here
// }

lis, err := net.Listen("tcp", ":8080")
if err != nil {
	log.Fatalf("Failed to listen: %v", err)
}
s := grpc.NewServer()
kafka.RegisterKafkaServiceServer(s, &kafkaServer{producer: producer})
if err := s.Serve(lis); err != nil {
	log.Fatalf("Failed to serve: %v", err)
}
