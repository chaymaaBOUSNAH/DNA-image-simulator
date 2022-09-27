import unittest
import sys
sys.path.append('./Biologic_Noise//utils_bio')
from Biologic_Noise.utils_bio import verify_csv

class BioTestCase(unittest.TestCase):

    def test_csv(self):
        # assert:
        csv_path = r"C:\Users\cbousnah\Desktop\GENERATOR\fibers_coords"
        image_file = 'image_1_mask.png'

        # when
        csv_data = verify_csv(csv_path, image_file)

        # then
        self.assertEqual(csv_data, 'csv data is extracted')


if __name__ == '__main__':
    unittest.main()
