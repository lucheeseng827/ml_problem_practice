from wx.lib.pubsub import pub


class MyPublisher:
    def __init__(self):
        self.publisher = pub.Publisher()

    def publish_event(self, data):
        self.publisher.sendMessage("my_event", data=data)


class MySubscriber:
    def __init__(self):
        pub.subscribe(self.on_my_event, "my_event")

    def on_my_event(self, data):
        print("Received event with data:", data)


# Create publisher and subscriber objects
publisher = MyPublisher()
subscriber = MySubscriber()

# Publish an event
publisher.publish_event("Hello, world!")
