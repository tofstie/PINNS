from src.dataloader.dataloader_factory import DataLoaderFactory

def main():
    dataloader = DataLoaderFactory.create_data_loader("category")

if __name__ == '__main__':
    main()