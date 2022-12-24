use rocket::Route;

#[put("/users/<id>", format = "application/json", data = "<user_data>")]
fn update_user(id: i32, user_data: Json<User>) -> Json<User> {
    // Validate and update user by ID
    let updated_user = User::update(id, user_data.into_inner());
    Json(updated_user)
}
