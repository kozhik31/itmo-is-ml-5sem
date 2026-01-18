from models import ImgDataset
from constants import *
from torch.utils.data import DataLoader


def create_dataloaders(X_train_values, y_train_values, train_transforms, test_trasforms,
                       X_test_values, y_test_values,
                       X_val_values, y_val_values,
                       batch_size=64):
    train_ds = ImgDataset(X_train_values, y_train_values, train_transforms)
    test_ds = ImgDataset(X_test_values, y_test_values, test_trasforms)
    val_ds = ImgDataset(X_val_values, y_val_values, test_trasforms)

    dataloaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    }

    return dataloaders
