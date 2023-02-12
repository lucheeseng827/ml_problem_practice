use rocksdb::{DB, Options, WriteOptions};

// Create a new RocksDB instance with a WAL
let mut opts = Options::default();
opts.create_if_missing(true);
let db = DB::open(opts, "path/to/db").unwrap();

// Start listening to changes in the source system
let changes = listen_for_changes();

// Write the changes to RocksDB
for change in changes {
    db.put(change.key, change.value, &WriteOptions::default());
}

// Close the RocksDB instance
db.close().unwrap();
