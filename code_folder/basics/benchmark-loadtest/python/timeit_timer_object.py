import timeit

code_snippet = """
result = 0
for i in range(100):
    result += i
"""

timer = timeit.Timer(code_snippet)
execution_time = timer.timeit(number=1000)
print(f"Execution time: {execution_time:.5f} seconds")
