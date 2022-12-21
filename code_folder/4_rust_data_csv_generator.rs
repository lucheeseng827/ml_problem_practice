// This code will spawn three threads, each of which generates data for a column in the CSV file. The threads send the data to the main thread using a channel, and the main thread writes the data to the file

use std::env;
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::mpsc;
use std::thread;

fn main() -> Result<(), Box<dyn Error>> {
    // Set up the channels for communication between threads
    let (tx, rx) = mpsc::channel();

    // Spawn a thread for each column
    for i in 0..3 {
        let tx = tx.clone();
        thread::spawn(move || {
            // Generate the data for the column
            let data: Vec<String> = (0..10).map(|j| format!("{},{}", i, j)).collect();

            // Send the data to the main thread
            tx.send((i, data)).unwrap();
        });
    }

    // Open the output file
    let mut file = BufWriter::new(File::create("output.csv")?);

    // Write the header row
    writeln!(file, "col1,col2,col3")?;

    // Receive the data from the threads and write it to the file
    for (i, data) in rx {
        for row in data {
            writeln!(file, "{}", row)?;
        }
    }

    Ok(())
}
