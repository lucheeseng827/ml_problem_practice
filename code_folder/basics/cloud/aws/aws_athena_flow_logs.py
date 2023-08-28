import boto3


def run_athena_query(query, database, output_bucket):
    client = boto3.client("athena")

    response = client.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={
            "OutputLocation": "s3://" + output_bucket,
        },
    )

    # Returning the query ID for later retrieval
    return response["QueryExecutionId"]


def get_query_results(query_id):
    client = boto3.client("athena")
    response = client.get_query_results(QueryExecutionId=query_id)

    # Extracting and returning the results
    return [item["Data"] for item in response["ResultSet"]["Rows"]]


if __name__ == "__main__":
    # Sample query
    query = "SELECT * FROM your_athena_table WHERE ... LIMIT 10;"
    database = "your_athena_database"
    output_bucket = "YOUR_BUCKET_NAME/path_for_query_results"

    query_id = run_athena_query(query, database, output_bucket)
    results = get_query_results(query_id)
    for row in results:
        print(row)
