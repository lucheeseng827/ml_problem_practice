use aws_sdk_s3::Client;
use avro_rs::{Reader, Writer};
use parquet::{
    file::writer::{FileWriter, RowGroupWriter},
    schema::{types::Type, SchemaDescriptor},
    util::memory::ByteBufferPtr,
};
use rusoto_core::{Region, RusotoError};
use rusoto_credential::StaticProvider;
use rusoto_s3::{
    GetObjectRequest, ListObjectsV2Request, PutObjectRequest, RusotoS3, S3Client, S3,
};
use std::{
    io::{BufReader, Read},
    sync::{Arc, Mutex},
    thread,
};

fn main() {
    let bucket_name = "my-bucket";
    let object_key = "data.txt";
    let parquet_output = "data.parquet";
    let avro_output = "data.avro";

    let access_key = "my-access-key";
    let secret_key = "my-secret-key";
    let region = Region::UsEast1;

    let provider = StaticProvider::new(access_key, secret_key, None, None);
    let s3_client = S3Client::new_with(
        rusoto_core::HttpClient::new().unwrap(),
        provider,
        region,
    );

    let get_obj_req = GetObjectRequest {
        bucket: bucket_name.to_string(),
        key: object_key.to_string(),
        ..Default::default()
    };

    let mut get_obj_output = match s3_client.get_object(get_obj_req).sync() {
        Ok(output) => output,
        Err(error) => {
            eprintln!("Failed to get object from S3: {:?}", error);
            return;
        }
    };

    let mut s3_obj_data = Vec::new();
    match get_obj_output.body.take() {
        Some(mut body) => {
            body.read_to_end(&mut s3_obj_data).unwrap();
        }
        None => {
            eprintln!("Failed to read object data from S3");
            return;
        }
    }

    let schema = Arc::new(SchemaDescriptor::new("data".to_string(), vec![
        Type::primitive_type(Default::default(), Type::INT32, "id".to_string()),
        Type::primitive_type(Default::default(), Type::STRING, "name".to_string()),
        Type::primitive_type(Default::default(), Type::FLOAT, "value".to_string())
    ]));

    let row_group_writer = {
        let file_writer = FileWriter::new(schema.clone(), ByteBufferPtr::new(vec![])).unwrap();
        file_writer.start_row_group().unwrap()
    };

    let mut writer = BufWriter::new(row_group_writer);
    for line in s3_obj_data.lines() {
        let mut fields = line.split('\t');
        let id = fields.next().unwrap().parse::<i32>().unwrap();
        let name = fields.next().unwrap().to_string();
        let value = fields.next().unwrap().parse::<f32>().unwrap();
        let row = vec![
            parquet::column::writer::get_typed_column_writer(&schema, "id", &mut writer)
                .unwrap()
                .write_batch(1, &[id])
                .unwrap(),
            parquet::column::writer::get_typed_column_writer(&schema, "name", &mut writer)
                .unwrap()
                .write_batch(1, &[name])
                .unwrap(),
            parquet::column::writer::get_typed_column_writer(&schema, "value", &mut writer)
                .unwrap()
                .write_batch(1, &[value])
