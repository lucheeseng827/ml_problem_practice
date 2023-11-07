import base64

import requests
from nacl import encoding, public


def encrypt(public_key: str, secret_value: str) -> str:
    """Encrypt a Unicode string using the public key."""
    public_key = public.PublicKey(public_key.encode("utf-8"), encoding.Base64Encoder())
    sealed_box = public.SealedBox(public_key)
    encrypted = sealed_box.encrypt(secret_value.encode("utf-8"))
    return base64.b64encode(encrypted).decode("utf-8")

# Personal access token from GitHub
token = 'your_github_token'
owner = 'repo_owner'
repo = 'repo_name'
secret_name = 'MY_SECRET'
secret_value = 'my_secret_value'

# Get the public key
url = f'https://api.github.com/repos/{owner}/{repo}/actions/secrets/public-key'
headers = {
    'Authorization': f'token {token}',
    'Accept': 'application/vnd.github.v3+json',
}
response = requests.get(url, headers=headers)
public_key = response.json()['key']
key_id = response.json()['key_id']

# Encrypt the secret
encrypted_value = encrypt(public_key, secret_value)

# Create or update the secret
url = f'https://api.github.com/repos/{owner}/{repo}/actions/secrets/{secret_name}'
data = {
    'encrypted_value': encrypted_value,
    'key_id': key_id,
}
response = requests.put(url, headers=headers, json=data)

print(f'Secret created: {response.status_code}')
