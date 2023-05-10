use warp::Filter;
use warp_plus::restful;

#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct Item {
    id: u32,
    name: String,
}

#[tokio::main]
async fn main() {
    let items = vec![
        Item { id: 1, name: "item1".into() },
        Item { id: 2, name: "item2".into() },
    ];

    let list_items = warp::path!("items")
        .and(warp::get())
        .map(move || warp::reply::json(&items));

    let get_item = warp::path!("items" / u32)
        .and(warp::get())
        .map(move |id| {
            for item in &items {
                if item.id == id {
                    return warp::reply::json(item);
                }
            }
            warp::reply::with_status(
                warp::reply::json(&"Item not found"),
                warp::http::StatusCode::NOT_FOUND,
            )
        });

    let routes = list_items
        .or(get_item)
        .with(warp::log("api"))
        .with(restful());

    warp::serve(routes).run(([127, 0, 0, 1], 3030)).await;
}
