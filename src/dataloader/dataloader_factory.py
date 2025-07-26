from src.dataloader.category_dataloader import DataLoaderCategory
from src.dataloader.philip_dataloader import DataLoaderPhilip


class DataLoaderFactory:
    def __init__(self):
        return

    @staticmethod
    def create_data_loader(data_loader_selection, test_size: float = 0.2, random_state: int = 1):
        if data_loader_selection == 'philip':
            return DataLoaderPhilip(test_size=test_size, random_state=random_state)
        elif data_loader_selection == 'category':
            return DataLoaderCategory(test_size=test_size, random_state=random_state)
        else:
            raise ValueError('Invalid data_loader_selection')
