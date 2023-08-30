import argparse

import apache_beam as beam
from apache_beam.io import ReadFromText, WriteToPubSub
from apache_beam.io.gcp.pubsub import PubsubMessage
from apache_beam.options.pipeline_options import PipelineOptions


class ConvertToPubSubMessage(beam.DoFn):
    def process(self, element):
        # Assuming each line in the GCS file is a separate message.
        return [PubsubMessage(element, attributes=None)]


def run(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input GCS path")
    parser.add_argument("--topic", dest="topic", help="Output Pub/Sub topic")
    known_args, pipeline_args = parser.parse_known_args(argv)

    options = PipelineOptions(pipeline_args)

    with beam.Pipeline(options=options) as p:
        # Read data from GCS.
        lines = p | "Read from GCS" >> ReadFromText(known_args.input)

        # Convert each line to a PubSub message.
        messages = lines | "Convert to PubSub Message" >> beam.ParDo(
            ConvertToPubSubMessage()
        )

        # Write to Pub/Sub.
        messages | "Write to PubSub" >> WriteToPubSub(topic=known_args.topic)


if __name__ == "__main__":
    run()


"""
to run this script:
python gcp_dataflow_gcs_puller_pubsub_write.py --input gs://<bucket_name>/<file_name> --topic projects/<project_name>/topics/<topic_name>
"""
