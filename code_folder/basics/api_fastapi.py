"""
$ pip install fastapi[all]
$ pip install uvicorn
$ pip install swagger-ui-bundle

# ASGI server implementation. Uvicorn is a lightning-fast ASGI server implementation, using uvloop and httptools.
uvicorn main:app --reload

https://fastapi.tiangolo.com/tutorial/openapi-schema/overview/

https://fastapi.tiangolo.com/tutorial/openapi-schema/additional-info/
"""

from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}


# Add this block to generate swagger documentation
openapi_prefix = "/openapi"


@app.get(openapi_prefix)
def get_openapi_route():
    return get_openapi(
        title="My API",
        version="1.0.0",
        description="This is my API",
        routes=app.routes,
    )


@app.get(openapi_prefix + "_html")
def get_openapi_html_route():
    return get_swagger_ui_html(openapi_url=openapi_prefix)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
