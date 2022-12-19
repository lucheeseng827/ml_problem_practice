use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use mysql::{Pool, Opts};

fn main() -> std::io::Result<()> {
    let opts = Opts::from_url("mysql://user:password@localhost:3306/test").unwrap();
    let pool = Pool::new(opts).unwrap();

    HttpServer::new(move || {
        App::new()
            .route("/users", web::get().to(users))
    })
    .bind("127.0.0.1:8080")?
    .run()
}

fn users() -> impl Responder {
    let mut conn = pool.get_conn().unwrap();
    let res = conn.query("SELECT * FROM users").unwrap();
    HttpResponse::Ok().json(res)
}
