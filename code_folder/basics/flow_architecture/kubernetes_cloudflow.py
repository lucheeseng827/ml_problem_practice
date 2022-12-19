from cloudflow.stream import Stream
from cloudflow.operator import (
    Operator,
    InputStream,
    OutputStream,
)

class MyOperator(Operator):
    def __init__(self, input_stream: InputStream, output_stream: OutputStream) -> None:
        super().__init__(input_stream, output_stream)

    def process_element(self, element: Any, timestamp: int) -> Optional[List[Any]]:
        # process the element and return a list of elements to write to the output stream
        return [processed_element]

# define the input and output streams
input_stream = Stream(name="input")
output_stream = Stream(name="output")

# create an instance of your operator and connect the streams
op = MyOperator(input_stream, output_stream)

# start the streaming application
app = Cloudflow([input_stream, output_stream])
app.start()
