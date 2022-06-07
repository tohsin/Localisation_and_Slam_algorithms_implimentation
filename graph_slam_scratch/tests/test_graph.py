
import sys

# adding Folder_2 to the system path
sys.path.insert(0, '/Users/emma/dev/Localisation_and_Slam_algorithms_implimentation/graph_slam_scratch')
import os
import unittest
from  graphSlam import graph as gf


class TestGraph(unittest.TestCase):
    def setUp(self) -> None:
        pass
        # return super().setUp()

    def test_optimise(self):
        g = gf.Graph()
        g.optimise()
        self.assertIsNotNone(g)
        # self.assertTrue(False)


if __name__ == "__main__":
    unittest.main()