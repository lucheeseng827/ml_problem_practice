import unittest
from lambda_function import lambda_handler

class TestLambdaFunction(unittest.TestCase):

    def test_lambda_handler(self):
        event = {
            'project_name': 'my_project',
            'owner_name': 'john_doe',
            'start_date': '2022-04-01',
            'end_date': '2022-04-30',
            'region': 'us-west-2'
        }
        context = None
        response = lambda_handler(event, context)
        self.assertEqual(response['status'], 'success')
        self.assertGreater(len(response['message']), 0)

if __name__ == '__main__':
    unittest.main()
