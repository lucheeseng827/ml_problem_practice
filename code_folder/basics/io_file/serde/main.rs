#[derive(Deserialize)]
struct Data {
    field1: String,
    field2: i32,
}


use std::fs::File;
use std::io::Read;

let mut file = File::open("path/to/file.txt")?;
let mut file_contents = String::new();
file.read_to_string(&mut file_contents)?;


use serde_json::from_str;

let data: Data = from_str(&file_contents)?;
