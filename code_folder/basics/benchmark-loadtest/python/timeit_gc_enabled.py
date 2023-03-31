import timeit
import gc

code_snippet = """
result = 0
for i in range(100):
    result += i
"""

# Enable garbage collection
gc.enable()

execution_time = timeit.timeit(code_snippet, number=1000)
print(f"Execution time: {execution_time:.5f} seconds")
