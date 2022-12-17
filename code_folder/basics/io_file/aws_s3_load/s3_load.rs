use std::error::Error;
use std::io::{Read, Write};
use aws_sdk_rust::s3::S3Client;
use aws_sdk_rust::s3::model::{GetObjectRequest, PutObjectRequest};

let s3_client = S3Client::new(Region::UsEast1);
let get_object_request = GetObjectRequest {
    bucket: "my-bucket".to_string(),
    key: "my-key".to_string(),
    ..Default::default()
};

let get_object_output = s3_client.get_object(get_object_request).await?;
let mut object_contents = Vec::new();
get_object_output.body.read_to_end(&mut object_contents).await?;
let put_object_request = PutObjectRequest {
    body: Some(object_contents.into()),
    bucket: "my-bucket".to_string(),
    key: "my-key".to_string(),
    ..Default::default()
};

s3_client.put_object(put_object_request).await?;
