import boto3

s3_client = boto3.client("s3")
response = s3_client.create_bucket(
    Bucket="boto-3-tutorial-bucket-1"
)
print(response)
