from torch.utils.data import DataLoader, Dataset, random_split


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    val_split: bool = False,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
):
    """
    Splits the dataset into train, test, and optionally validation sets, and returns DataLoaders.

    Args:
        dataset (Dataset): The dataset to split.
        batch_size (int): The batch size for the DataLoaders.
        shuffle (bool): Whether to shuffle the dataset before splitting.
        val_split (bool): Whether to create a validation set.
        train_ratio (float): The proportion of data to use for training (default: 0.8).
        val_ratio (float): The proportion of data to use for validation if val_split is True (default: 0.1).

    Returns:
        dict: A dictionary containing DataLoaders for train, test, and optionally validation.
    """
    dataset_size = len(dataset)

    if val_split:
        train_size = int(train_ratio * dataset_size)
        val_size = int(val_ratio * dataset_size)
        test_size = dataset_size - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
    else:
        train_size = int(train_ratio * dataset_size)
        test_size = dataset_size - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        val_dataset = None

    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    }

    if val_split:
        dataloaders["val"] = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

    return dataloaders
