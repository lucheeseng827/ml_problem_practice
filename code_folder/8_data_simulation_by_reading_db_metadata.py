# python create data set with pandas which is read on mysql code
# Path: 9_data_simulation_by_reading_db_metadata.py

import pandas as pd
import pymysql

# create connection
connection = pymysql.connect(
    host="localhost", passwd="password", user="root", db="test"
)

# create cursor
cursor = connection.cursor()

# execute query
data = cursor.execute("SELECT col1,col2,col3 FROM test_table")

# fetch data
data = cursor.fetchall()

# make it more customizable


def create_data_set(number_of_rows, number_of_columns):
    # Create a dataframe with 1000 rows and 3 columns
    df = pd.DataFrame(
        number_of_rows * [range(number_of_columns)],
        columns=["feature_1", "feature_2", "feature_3"],
    )

    return df


data_simulate = create_data_set(1000, 3)
