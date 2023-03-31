import timeit

def my_function():
    result = 0
    for i in range(100):
        result += i

# Measure the execution time of the function
execution_time = timeit.timeit(my_function, number=1000)
print(f"Execution time: {execution_time:.5f} seconds")
