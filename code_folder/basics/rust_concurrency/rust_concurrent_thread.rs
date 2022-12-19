use std::thread;

fn main() {
    let handle = thread::spawn(|| {
        // Perform task in separate thread
    });
    handle.join().unwrap();
}
