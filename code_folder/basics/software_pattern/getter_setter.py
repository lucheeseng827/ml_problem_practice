class Employee:
    def __init__(self, name, salary):
        self._name = name
        self._salary = salary

    def get_name(self):
        return self._name

    def set_name(self, name):
        self._name = name

    def get_salary(self):
        return self._salary

    def set_salary(self, salary):
        self._salary = salary

# Use the getter and setter methods
employee = Employee("John Smith", 50000)
print(employee.get_name())  # prints "John Smith"
employee.set_salary(60000)
print(employee.get_salary())  # prints 60000
