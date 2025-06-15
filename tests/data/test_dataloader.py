from project_name.data.data_loader import DataLoader
import unittest


class TestDataloader(unittest.TestCase):
    def setUp(self):
        self.dataloader = DataLoader("val")

    def test_index(self):
        current_index = self.dataloader.data_index
        self.dataloader.increment_index()
        self.assertEqual((current_index + 1) % len(self.dataloader.data_paths),
                         self.dataloader.data_index)

    def test_data_getting(self):
        data = self.dataloader.get_data()
        self.assertEqual(len(data), 2)


if __name__ == '__main__':
    unittest.main()
