import sodasql

# create a Soda SQL configuration object
config = {
    "type": "postgresql",
    "creds": {
        "host": "localhost",
        "port": 5432,
        "database": "my_database",
        "username": "my_username",
        "password": "my_password"
    }
}

# create a Soda SQL context object
context = sodasql.contexts.DbContext(config)

# create a Soda SQL validator object
validator = sodasql.validators.SqlAlchemyValidator(context)

# define a data validation test
test_query = """
SELECT COUNT(*) AS total_records
FROM my_table
WHERE my_column IS NULL;
"""

# run the validation test and get the results
results = validator.execute_test(test_query)

# print the validation results
print(results.to_json())
