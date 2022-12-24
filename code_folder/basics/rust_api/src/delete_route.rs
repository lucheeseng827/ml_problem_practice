use rocket::Route;

#[delete("/users/<id>")]
fn delete_user(id: i32) -> Json<User> {
    // Delete user by ID
    let deleted_user = User::delete(id);
    Json(deleted_user)
}
