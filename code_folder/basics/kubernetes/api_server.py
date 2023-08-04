import requests


def check_api_status(api_url, api_key):
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()  # Raise an exception for any HTTP errors

        # Check the response status code or any other relevant data
        if response.status_code == 200:
            print("API server is online and accessible.")
        else:
            print("API server is online, but the response indicates an error.")

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        print("API server is unreachable.")


# Usage example
api_url = "https://api.example.com/status"  # Replace with the actual API URL
api_key = "your_api_key"  # Replace with your API key
check_api_status(api_url, api_key)
