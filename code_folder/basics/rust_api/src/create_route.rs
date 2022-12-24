use rocket::Route;

#[post("/users", format = "application/json", data = "<user_data>")]
fn create_user(user_data: Json<User>) -> Json<User> {
    // Validate and create new user
    let new_user = User::create(user_data.into_inner());
    Json(new_user)
}
