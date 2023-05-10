import requests

# Define the base URL of the Flask application
BASE_URL = "http://localhost:5000"


def test_home_page():
    # Send a GET request to the home page
    response = requests.get(BASE_URL)

    # Check if the response status code is 200 (OK)
    assert response.status_code == 200

    # Check if the response contains the expected text
    assert "Welcome to the Flask App" in response.text


def test_about_page():
    # Send a GET request to the about page
    response = requests.get(BASE_URL + "/about")

    # Check if the response status code is 200 (OK)
    assert response.status_code == 200

    # Check if the response contains the expected text
    assert "This is the about page" in response.text


# Run the smoke tests
test_home_page()
test_about_page()
