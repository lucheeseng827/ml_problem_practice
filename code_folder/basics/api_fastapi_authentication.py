""" One way to implement authentication in a FastAPI application is to use the FastAPI-Security library, which provides several pre-built security schemes.

Here is an example of how you can use FastAPI-Security to implement JWT (JSON Web Token) authentication in a FastAPI application:
In this example, we define an /auth/token endpoint that authenticates the user based on the provided username and password. If the authentication is successful, it returns a JWT token.

We also define a protected endpoint /items/, which requires the me scope to access. The SecurityScopes class is used to check if the required scopes are present in the JWT token.

You can find more information about FastAPI-Security and other ways to implement authentication in FastAPI in the documentation: https://fastapi-security.tiangolo.com/"""

from fastapi import FastAPI
from fastapi_security import (
    OAuth2PasswordBearer,
    SecurityScopes,
    security_scheme_generator,
)

app = FastAPI()

security_schemes = security_scheme_generator(
    {
        "bearer": OAuth2PasswordBearer(
            tokenUrl="/auth/token",
            scopes={"me": "Read and write access to your own data"}
        )
    }
)

@app.post("/auth/token")
async def login(username: str, password: str, scopes: SecurityScopes):
    if username != "test" or password != "test":
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    return {"access_token": "fake-jwt-token", "token_type": "bearer"}

@app.get("/items/")
async def read_items(scopes: SecurityScopes = SecurityScopes(scopes={"me"})):
    return [{"item_id": "Foo"}]
