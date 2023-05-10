from flask import Flask
from flask_restplus import Api, Resource

app = Flask(__name__)

# Initialize Flask-RESTPlus extension with Flask app
api = Api(
    app,
    version="1.0",
    title="Sample API",
    description="A simple demo API",
    doc="/api/swagger/",
)

# Create a namespace for the API
ns = api.namespace("items", description="Items operations")

# Sample data
ITEMS = [{"id": 1, "name": "item1"}, {"id": 2, "name": "item2"}]


@ns.route("/")  # Define route within the namespace
class ItemList(Resource):
    @ns.doc("list_items")
    def get(self):
        """List all items"""
        return ITEMS


@ns.route("/<int:id>")
@ns.param("id", "The item identifier")
@ns.response(404, "Item not found")
class Item(Resource):
    @ns.doc("get_item")
    def get(self, id):
        """Fetch a item given its identifier"""
        for item in ITEMS:
            if item["id"] == id:
                return item
        api.abort(404, f"Item {id} doesn't exist")


if __name__ == "__main__":
    app.run(debug=True)
