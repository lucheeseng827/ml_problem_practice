# Secret generation on github via API

retrieve the public key

```bash
GITHUB_TOKEN="your_github_token"
OWNER="repo_owner"
REPO="repo_name"

# Get the public key
PUBLIC_KEY=$(curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/repos/$OWNER/$REPO/actions/secrets/public-key)

# Parse the key and key_id
export PUBLIC_KEY=$(echo $PUBLIC_KEY | jq -r '.key')
export KEY_ID=$(echo $PUBLIC_KEY | jq -r '.key_id')
```

encrypt the secret

```bash
# Your secret value
SECRET_VALUE="my_secret_value"

# Encrypt your secret using OpenSSL and the public key
ENCRYPTED_VALUE=$(echo -n $SECRET_VALUE | openssl enc -aes-256-cbc -e -A -base64 -K $PUBLIC_KEY)
```


Update the secret in the repository

```bash
# The name of your secret
SECRET_NAME="MY_SECRET"

curl -X PUT -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github.v3+json" \
    https://api.github.com/repos/$OWNER/$REPO/actions/secrets/$SECRET_NAME \
    -d @- << EOF
{
  "encrypted_value": "$ENCRYPTED_VALUE",
  "key_id": "$KEY_ID"
}
EOF
```
