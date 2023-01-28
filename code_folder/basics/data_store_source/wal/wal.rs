use walrus::Wal;

// Create a new WAL
let mut wal = Wal::create("path/to/wal").unwrap();

// Write a value to the WAL
wal.write(b"my_key", b"my_value").unwrap();

// Read a value from the WAL
let value = wal.read(b"my_key").unwrap();

// Close the WAL
wal.close().unwrap();
