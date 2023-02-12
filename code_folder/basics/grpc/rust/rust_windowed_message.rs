use futures::StreamExt;
use tonic::{transport::Server, Request, Response, Status};

// Define the protobuf message structure
#[derive(Default, Debug, Clone, PartialEq, Message)]
pub struct Event {
    pub id: i32,
    pub data: String,
}

// Define the gRPC service
pub trait EventService {
    async fn stream_events(
        &self,
        request: Request<tonic::Streaming<Event>>,
    ) -> Result<Response<tonic::Streaming<Event>>, Status>;
}

// Implement the gRPC service
#[derive(Debug, Default, Clone)]
pub struct MyEventService;

#[tonic::async_trait]
impl EventService for MyEventService {
    async fn stream_events(
        &self,
        request: Request<tonic::Streaming<Event>>,
    ) -> Result<Response<tonic::Streaming<Event>>, Status> {
        let mut events = request.into_inner();
        while let Some(event) = events.next().await {
            let event = event?;
            println!("Received event: {:?}", event);
        }
        Ok(Response::new(tonic::Streaming::new(Vec::new())))
    }
}

// Start the gRPC server
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "[::1]:50051".parse().unwrap();
    let event_service = MyEventService::default();

    println!("gRPC server listening on {}", addr);

    Server::builder()
        .add_service(EventServiceServer::new(event_service))
        .serve(addr)
        .await?;

    Ok(())
}
