// create file based on data read from stdin
//
// https://doc.rust-lang.org/std/io/struct.Stdin.html
// https://doc.rust-lang.org/std/io/trait.Read.html
// https://doc.rust-lang.org/std/io/trait.Write.html
// https://doc.rust-lang.org/std/io/trait.Seek.html
// https://doc.rust-lang.org/std/io/trait.BufRead.html
// https://doc.rust-lang.org/std/io/trait.BufRead.html#tymethod.read_line
// https://doc.rust-lang.org/std/io/trait.BufRead.html#tymethod.read_until
// https://doc.rust-lang.org/std/io/trait.BufRead.html#tymethod.read_to_string
// https://doc.rust-lang.org/std/io/trait.BufRead.html#tymethod.read_to_end
// https://doc.rust-lang.org/std/io/trait.BufRead.html#tymethod.lines
// https://doc.rust-lang.org/std/io/trait.BufRead.html#tymethod.split
// https://doc.rust-lang.org/std/io/trait.BufRead.html#tymethod.splitn


fn main {
    let mut input = String::new();
    let mut file = File::create("output.txt").unwrap();
    let mut stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        input.clear();
        stdout.write(b"Enter text: ").unwrap();
        stdout.flush().unwrap();
        stdin.read_line(&mut input).unwrap();
        if input.trim() == "quit" {
            break;
        }
        file.write(input.as_bytes()).unwrap();
    }
}
