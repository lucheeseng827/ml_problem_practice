import qrcode
import boto3
import urllib.request

s3 = boto3.client('s3')

def lambda_handler(event, context):
    # Retrieve the URL from the event
    url = event['url']

    # Generate the QR code image from the URL
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=4)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")

    # Save the image to a temporary file
    img_path = '/tmp/qrcode.png'
    img.save(img_path)

    # Upload the image to S3
    bucket_name = 'my-bucket'
    key = 'qrcodes/{}.png'.format(url.split('/')[-1])
    s3.upload_file(img_path, bucket_name, key)

    return {
        'statusCode': 200,
        'body': 'QR code image uploaded to S3'
    }
