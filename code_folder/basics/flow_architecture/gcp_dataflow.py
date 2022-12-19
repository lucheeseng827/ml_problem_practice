import dataplex

# define the pipeline
pipeline = dataplex.Pipeline()

# read data from a CSV file
source = dataplex.CSVFileSource("input.csv")

# process the data
processor = dataplex.Processor(process_fn)

# write the results to a database
sink = dataplex.DatabaseSink("database_url")

# connect the source, processor, and sink in a pipeline
pipeline.source(source).processor(processor).sink(sink)

# run the pipeline
pipeline.run()
