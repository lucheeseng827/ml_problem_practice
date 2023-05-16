use rocket::response::content::Html;


#[macro_use]
extern crate rocket;

#[get("/")]
fn index() -> &'static str {
    "Hello, World!"
}

#[get("/admin")]
fn admin() -> &'static str {
    "Admin Page"
}

#[launch]
fn rocket() -> _ {
    rocket::build().
    mount("/", routes![index])
    .mount("/admin", routes![admin])
    .mount("/static", rocket::fs::dir("static"))
    .attach(Template::fairing())
}




#[get("/admin")]
fn admin() -> Html<&'static str> {
    Html(r#"<!DOCTYPE html>
        <html>
        <head>
            <title>Admin Page</title>
        </head>
        <body>
            <h1>Welcome to the Admin Page</h1>
            <!-- Add your access control form and UI elements here -->
        </body>
        </html>"#)
}
