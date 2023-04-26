import unittest
from unittest.mock import MagicMock
from botocore.exceptions import ClientError
from my_lambda_function import delete_unused_ebs

class TestDeleteUnusedEbs(unittest.TestCase):

    def setUp(self):
        self.ec2_mock = MagicMock()
        self.volume1 = {
            'VolumeId': 'vol-123',
            'Tags': {
                'Name': 'my-volume',
                'Owner': 'anansi'
            }
        }
        self.volume2 = {
            'VolumeId': 'vol-456',
            'Tags': {
                'Name': 'another-volume',
                'Owner': 'anansi'
            }
        }
        self.volume3 = {
            'VolumeId': 'vol-789',
            'Tags': {
                'Name': 'unused-volume',
                'Owner': 'anansi'
            }
        }

    def test_delete_unused_ebs(self):
        self.ec2_mock.describe_volumes.return_value = {
            'Volumes': [self.volume1, self.volume2, self.volume3]
        }
        delete_unused_ebs('Owner', 'anansi', self.ec2_mock)
        self.ec2_mock.delete_volume.assert_has_calls([
            MagicMock(VolumeId='vol-789'),
        ])

    def test_delete_unused_ebs_with_error(self):
        self.ec2_mock.describe_volumes.side_effect = ClientError({'Error': {}}, 'operation')
        with self.assertRaises(ClientError):
            delete_unused_ebs('Owner', 'anansi', self.ec2_mock)
