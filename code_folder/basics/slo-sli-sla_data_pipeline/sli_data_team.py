import datetime


def estimate_sli(data_process_time, data_arrival_time, required_processing_time):
    processing_delay = data_process_time - data_arrival_time
    sli_percentage = (
        (required_processing_time - processing_delay) / required_processing_time * 100
    )

    return sli_percentage


# Example usage
data_process_time = datetime.datetime(
    2023, 7, 15, 10, 30, 0
)  # Replace with the actual data process time
data_arrival_time = datetime.datetime(
    2023, 7, 15, 10, 0, 0
)  # Replace with the actual data arrival time
required_processing_time = datetime.timedelta(
    hours=1
)  # Replace with the actual required processing time

sli = estimate_sli(data_process_time, data_arrival_time, required_processing_time)
print(f"SLI: {sli}%")
