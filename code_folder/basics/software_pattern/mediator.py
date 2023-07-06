class Aircraft:
    def __init__(self, name, mediator):
        self.name = name
        self.mediator = mediator

    def send_message(self, message):
        self.mediator.send_message(message, self)

    def receive_message(self, message):
        print(f"{self.name} received message: {message}")


class AirTrafficController:
    def __init__(self):
        self.aircrafts = []

    def register_aircraft(self, aircraft):
        self.aircrafts.append(aircraft)

    def send_message(self, message, sender):
        for aircraft in self.aircrafts:
            if aircraft != sender:
                aircraft.receive_message(message)


# Create aircrafts and mediator (air traffic controller)
controller = AirTrafficController()
aircraft1 = Aircraft("Flight 1", controller)
aircraft2 = Aircraft("Flight 2", controller)
aircraft3 = Aircraft("Flight 3", controller)

# Register aircrafts with the controller
controller.register_aircraft(aircraft1)
controller.register_aircraft(aircraft2)
controller.register_aircraft(aircraft3)

# Aircrafts communicate through the mediator
aircraft1.send_message("Hello from Flight 1!")
aircraft2.send_message("Greetings from Flight 2!")
aircraft3.send_message("Message from Flight 3")
