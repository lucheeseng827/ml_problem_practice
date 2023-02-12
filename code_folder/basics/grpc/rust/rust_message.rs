use prost::Message;
use tonic::{transport::Server, Request, Response, Status};

// Define the protobuf message structure
#[derive(Default, Debug, Clone, PartialEq, Message)]
pub struct Greeting {
    pub name: String,
    pub message: String,
}

// Define the gRPC service
pub trait GreeterService {
    async fn say_hello(
        &self,
        request: Request<Greeting>,
    ) -> Result<Response<Greeting>, Status>;
}

// Implement the gRPC service
#[derive(Debug, Default, Clone)]
pub struct MyGreeterService;

#[tonic::async_trait]
impl GreeterService for MyGreeterService {
    async fn say_hello(
        &self,
        request: Request<Greeting>,
    ) -> Result<Response<Greeting>, Status> {
        let greeting = request.into_inner();
        println!("Received greeting: {:?}", greeting);
        Ok(Response::new(Greeting {
            name: "Response".to_owned(),
            message: format!("Hello, {}!", greeting.name),
        }))
    }
}

// Start the gRPC server
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "[::1]:50051".parse().unwrap();
    let greeter_service = MyGreeterService::default();

    println!("gRPC server listening on {}", addr);

    Server::builder()
        .add_service(GreeterServiceServer::new(greeter_service))
        .serve(addr)
        .await?;

    Ok(())
}
