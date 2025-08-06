import unittest

import numpy as np

from src.dataloader.dataloader_factory import DataLoaderFactory
from src.dataloader.pendulum_dataloader import DataLoaderPendulum
from src.dataloader.category_dataloader import DataLoaderCategory
from src.dataloader.philip_dataloader import DataLoaderPhilip

class DataLoaderTestCase(unittest.TestCase):
    def setUp(self):
        """
        Method that runs before every test
        """
        self.x_data = np.array([[2.,0.],[1.,2.],[1.5,1.],[1.2,0.8],[1.3,1.8]])
        self.y_data = np.array([1.,0.9,1.1,1.2,0.8])

        self.random_state = 42
        self.test_size = 0.2
    def _test_loader(self, data_loader_string):
        data_loader = DataLoaderFactory().create_data_loader(data_loader_string,test_size=self.test_size,random_state=self.random_state)
        data_loader.load_data(self.x_data, self.y_data)

        x_training, y_training = data_loader.get_train_data()
        x_test, y_test = data_loader.get_test_data()

        # Check is Scaler is working as expected
        self.assertEqual(float(data_loader.scaler.mean_[0]), 1.5)
        # Check to see if x data is changed
        self.assertEqual(float(x_training[1,0]),0.0)
        # Check to see y unchanged
        self.assertEqual(y_test[0], 0.9)
    def test_pendulum_loader(self):
        """
        Method called to test the pendulum data loader
        """
        data_loader_string = "pendulum"
        self._test_loader(data_loader_string)

    def test_category_loader(self):
        """
        Method called to test the category data loader
        """
        data_loader_string = "category"
        self._test_loader(data_loader_string)





if __name__ == '__main__':
    unittest.main(verbosity=1)
