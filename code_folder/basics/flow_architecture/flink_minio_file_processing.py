from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment

env = StreamExecutionEnvironment.get_execution_environment()
t_env = StreamTableEnvironment.create(env)

# MinIO settings
minio_bucket = "your-minio-bucket"
minio_endpoint = "minio-service:9000"
minio_access_key = "your-minio-access-key"
minio_secret_key = "your-minio-secret-key"

# Flink job
t_env.execute_sql(f"""
    CREATE TABLE source_table (
        word STRING
    ) WITH (
        'connector' = 'filesystem',
        'path' = 's3a://{minio_access_key}:{minio_secret_key}@{minio_endpoint}/{minio_bucket}',
        'format' = 'csv'
    )
""")

t_env.execute_sql("""
    CREATE TABLE result_table (
        word STRING,
        word_count BIGINT
    ) WITH (
        'connector' = 'print'
    )
""")

t_env.from_path("source_table") \
    .group_by("word") \
    .select("word, COUNT(word) as word_count") \
    .execute_insert("result_table")

env.execute("Word Count Job")
