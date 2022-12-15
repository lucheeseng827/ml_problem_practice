"""In this example, we create an IntSlider widget with a range of 1-10 and a step size of 2. We then display the widget using the display function, which allows us to interact with it in the Jupyter notebook or other Python environment.

You can also use the on_change method of the IntSlider widget to define a callback function that is called whenever the value of the widget is changed. This can be useful for performing some action in response to the user changing the value of the slider. Here is an example of how to use the on_change method to define a callback function:"""

from ipywidgets import IntSlider

# Create an IntSlider widget with a range of 1-10 and a step size of 2
slider = IntSlider(min=1, max=10, step=2)

# Define a callback function that prints the value of the slider
def on_value_change(change):
  print(change['new'])

# Register the callback function to be called when the value of the slider is changed
slider.observe(on_value_change, names='value')

# Display the widget
slider
