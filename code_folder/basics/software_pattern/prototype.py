import copy


class Prototype:
    def clone(self):
        return copy.deepcopy(self)


class ConcretePrototype(Prototype):
    def __init__(self, name):
        self.name = name

    def display(self):
        print(f"ConcretePrototype: {self.name}")


# Create an instance of the prototype
prototype = ConcretePrototype("Prototype")

# Clone the prototype to create new instances
clone1 = prototype.clone()
clone2 = prototype.clone()

# Display the cloned instances
clone1.display()  # Output: ConcretePrototype: Prototype
clone2.display()  # Output: ConcretePrototype: Prototype
