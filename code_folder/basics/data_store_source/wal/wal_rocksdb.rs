use rocksdb::{DB, Options, WriteOptions};

// Create a new RocksDB instance with a WAL
let mut opts = Options::default();
opts.create_if_missing(true);
opts.set_wal_dir("path/to/wal");
let db = DB::open(opts, "path/to/db").unwrap();

// Start a write transaction
let txn = db.transaction();

// Write a value to the WAL
txn.put(b"my_key", b"my_value", &WriteOptions::default());

// Commit the transaction
txn.commit().unwrap();

// Close the RocksDB instance
db.close().unwrap();
