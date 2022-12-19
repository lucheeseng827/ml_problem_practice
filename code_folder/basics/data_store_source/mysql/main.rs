use mysql::{Pool, Opts};

fn main() {
    let opts = Opts::from_url("mysql://user:password@localhost:3306/test").unwrap();
    let pool = Pool::new(opts).unwrap();

    let mut conn = pool.get_conn().unwrap();
    let res = conn.query("SELECT * FROM users").unwrap();
    println!("{:?}", res);
}
