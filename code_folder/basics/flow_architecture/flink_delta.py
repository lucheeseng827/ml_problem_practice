from flink.functions.GroupReduceFunction import GroupReduceFunction
from flink.plan.Environment import create_environment
from pyflink.dataset import ExecutionEnvironment
from pyflink.table import BatchTableEnvironment, DataTypes, TableConfig

# create a batch execution environment
env = ExecutionEnvironment.get_execution_environment()

# create a table environment
t_config = TableConfig()
t_env = BatchTableEnvironment.create(env, t_config)

# register a Delta table
t_env.connect(
    "delta",  # file system to read the delta table from
    {"path": "path/to/delta/table", "format": "delta"},
)

# read the table
table = t_env.from_path("delta.`path/to/delta/table`")

# define a custom function to process the table


class MyProcessFunction(GroupReduceFunction):
    def reduce(self, iterator, collector):
        # do some processing on the iterator and collect the results
        # ...
        # apply the custom function on the table
        args = [DataTypes.STRING()]
        result_type = DataTypes.STRING()
        result_table = table.group_by("column_name").reduce(
            MyProcessFunction(), *args, result_type=result_type
        )


result_table = table.group_by("column_name").reduce(MyProcessFunction())

# convert the result table to a DataSet and print it
result_dataset = t_env.to_dataset(result_table)
result_dataset.print()

# execute the Flink job
env.execute("My Flink Job")
