import os
import ssl
from OpenSSL import crypto
from flask import Flask

app = Flask(__name__)

# Function to create a self-signed SSL certificate
def generate_self_signed_cert(cert_file, key_file):
    k = crypto.PKey()
    k.generate_key(crypto.TYPE_RSA, 4096)

    cert = crypto.X509()
    cert.get_subject().CN = "localhost"  # Common Name (change as needed for your domain)
    cert.set_serial_number(1000)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(365 * 24 * 60 * 60)  # 1 year validity

    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(k)
    cert.sign(k, "sha256")

    with open(cert_file, "wt") as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert).decode("utf-8"))

    with open(key_file, "wt") as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k).decode("utf-8"))

# Check if the certificate and key files exist, if not generate them
cert_file = "certificate.pem"
key_file = "private_key.pem"

if not (os.path.exists(cert_file) and os.path.exists(key_file)):
    generate_self_signed_cert(cert_file, key_file)

# Load SSL context
context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
context.load_cert_chain(cert_file, key_file)

# Define your Flask routes as usual
@app.route("/")
def index():
    return "Hello, this is a secure Flask app served over HTTPS!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=443, ssl_context=context)
