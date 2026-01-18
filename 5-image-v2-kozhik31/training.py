import torch
import numpy as np
from tqdm import tqdm


device = 'cuda'

def fit(model, criterion, optimizer, scheduler, dataloaders, epoches=20):
    train_losses = []
    val_losses = []
    train_mae = []
    val_mae = []
    model = model.to(device)

    for epoch in range(epoches):
        print(f"\nEpoch {epoch + 1}/{epoches}")

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_mae = 0.0
            for images, targets in tqdm(dataloaders[phase]):
                images = images.to(device)
                targets = targets.unsqueeze(1).to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    mae = torch.mean(torch.abs(outputs - targets))

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * images.size(0)
                running_mae += mae.item() * images.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_mae = running_mae / len(dataloaders[phase].dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} | MAE: {epoch_mae:.2f} kcal")

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_mae.append(epoch_mae)
            else:
                val_losses.append(epoch_loss)
                val_mae.append(epoch_mae)
                scheduler.step(epoch_loss)

    return train_losses, val_losses, train_mae, val_mae

def test(model, criterion,dataloaders):
    model.eval()
    test_loss = 0.0
    test_mae = 0.0

    with torch.no_grad():
        for images, targets in dataloaders['test']:
            images = images.to(device)
            targets = targets.unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)
            mae = torch.mean(torch.abs(outputs - targets))

            test_loss += loss.item() * images.size(0)
            test_mae += mae.item() * images.size(0)

    test_loss /= len(dataloaders['test'].dataset)
    test_mae /= len(dataloaders['test'].dataset)

    print(f"TEST Loss: {test_loss:.2f} | TEST MAE: {test_mae:.2f} kcal")


def embeds(model, embed_loader):
    model.eval()
    all_embeds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in embed_loader:
            images = images.to(device)
            emb = model.get_embedding(images)
            all_embeds.append(emb.cpu().numpy())
            all_targets.append(targets.numpy())

    all_embeds = np.concatenate(all_embeds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    return all_embeds, all_targets