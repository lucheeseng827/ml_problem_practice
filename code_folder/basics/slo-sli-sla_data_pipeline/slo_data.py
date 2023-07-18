import datetime

def estimate_slo(data_process_time, data_arrival_time, required_processing_time):
    processing_delay = data_process_time - data_arrival_time
    processing_slo = required_processing_time - processing_delay

    if processing_slo > 0:
        slo_percentage = (processing_slo / required_processing_time) * 100
        return slo_percentage
    else:
        return 0

# Example usage
data_process_time = datetime.datetime(2023, 7, 15, 10, 30, 0)  # Replace with the actual data process time
data_arrival_time = datetime.datetime(2023, 7, 15, 10, 0, 0)  # Replace with the actual data arrival time
required_processing_time = datetime.timedelta(hours=1)  # Replace with the actual required processing time

slo = estimate_slo(data_process_time, data_arrival_time, required_processing_time)
print(f"SLO: {slo}%")
