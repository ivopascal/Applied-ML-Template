import unittest
from main import hello_world

# run all tests python -m unittest discover tests

class MainTest(unittest.TestCase):
    def test_hello(self):
        self.assertEqual(hello_world(), "Hello, World!")


if __name__ == '__main__':
    unittest.main()
