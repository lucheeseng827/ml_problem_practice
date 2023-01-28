mod create_route;
mod delete_route;
mod read_route;
mod update_route;

fn main() {
    create_route::create_user();
    read_route::read_user();
    update_route::update_user();
    delete_route::delete_user();
}
