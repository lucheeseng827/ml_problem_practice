import copy


class DocumentPrototype:
    def clone(self):
        return copy.deepcopy(self)

    def render(self):
        raise NotImplementedError


class Invoice(DocumentPrototype):
    def __init__(self, customer_name, amount):
        self.customer_name = customer_name
        self.amount = amount

    def render(self):
        print("Invoice")
        print(f"Customer: {self.customer_name}")
        print(f"Amount: ${self.amount}")
        print("---")


class Contract(DocumentPrototype):
    def __init__(self, client_name, terms):
        self.client_name = client_name
        self.terms = terms

    def render(self):
        print("Contract")
        print(f"Client: {self.client_name}")
        print("Terms:")
        for term in self.terms:
            print(f"- {term}")
        print("---")


# Create prototypes for invoice and contract
invoice_prototype = Invoice("John Doe", 1000)
contract_prototype = Contract("ABC Company", ["Term 1", "Term 2", "Term 3"])

# Clone prototypes to create new instances
invoice1 = invoice_prototype.clone()
invoice2 = invoice_prototype.clone()
contract1 = contract_prototype.clone()

# Customize cloned instances
invoice1.amount = 1500
contract1.terms.append("Term 4")

# Render the documents
invoice1.render()
invoice2.render()
contract1.render()
