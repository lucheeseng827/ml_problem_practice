import unittest
import requests


class TestAPI(unittest.TestCase):
    def test_get(self):
        r = requests.get("http://localhost:5000/users")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"users": ["John", "Jane"]})

    def test_post(self):
        r = requests.post("http://localhost:5000/users", json={"name": "Bob"})
        self.assertEqual(r.status_code, 201)


if __name__ == "__main__":
    unittest.main()
