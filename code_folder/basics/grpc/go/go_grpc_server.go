

lis, err := net.Listen("tcp", ":8080")
if err != nil {
	log.Fatalf("Failed to listen: %v", err)
}
s := grpc.NewServer()
kafka.RegisterKafkaServiceServer(s, &kafkaServer{producer: producer})
if err := s.Serve(lis); err != nil {
	log.Fatalf("Failed to serve: %v", err)
}
