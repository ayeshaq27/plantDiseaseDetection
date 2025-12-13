import torch

def train_one_epoch(model, dataloader, optimizer, criterion, device="cpu"):
    """
    Trains the model for one full epoch.
    
    Returns:
        avg_loss (float): average training loss
        accuracy (float): training accuracy
    """

    model.train()  # set model to training mode
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # 1. Forward pass
        outputs = model(images)

        # 2. Compute loss
        loss = criterion(outputs, labels)

        # 3. Zero gradients
        optimizer.zero_grad()

        # 4. Backpropagation
        loss.backward()

        # 5. Update model weights
        optimizer.step()

        running_loss += loss.item()

        # Compute accuracy
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy



def validate(model, dataloader, criterion, device="cpu"):
    """
    Evaluates the model on the validation dataset.
    
    Returns:
        avg_loss (float): average validation loss
        accuracy (float): validation accuracy
    """

    model.eval()  # evaluation mode (no dropout, no batchnorm updates)
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # no gradient calculation needed
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy
