# Import all the packages
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.optim as optim
import os
from tqdm import tqdm
from unet_utils  import UNet, CustomDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

# load current model to train
def load_model_and_optimizer(model, optimizer, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] 
        return start_epoch
    else:
        print(f"No checkpoint found at {checkpoint_path}. Starting training from scratch.")
        return 0  


# training model
def train(model, num_epochs, train_loader, optimizer, scheduler, start_epoch):
  epoch_iterator = tqdm(range(start_epoch, num_epochs), desc="Epochs")
  train_losses = []
  for epoch in epoch_iterator:
    batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for count, (x, y, filename) in enumerate(batch_iterator):
      model.train()
      x = x.to(device)
      y = y.to(device)
      out = model(x)
      out = torch.sigmoid(out)
      # loss = loss_function(out, y)
      loss = combined_loss(out, y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      train_losses.append(loss.item())

    val_loss = eval(model, val_loader, epoch)
    scheduler.step(val_loss)
    path = f'unet_model_{epoch + 1}.pt'
    torch.save(
      {
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'epoch' : epoch
      }
    , path)
    print(f"Model saved to {path}")

# evaluate model with testing data after each epoch

def eval(model, val_loader, epoch):
  model.eval()
  num_correct = 0
  num_pixels = 0
  total_loss = 0
  with torch.no_grad():
    val_iterator = tqdm(val_loader, desc=f"Evaluating Epoch {epoch + 1}")
    for x, y, filename in val_iterator:
      x = x.to(device)
      y = y.to(device)
      out_img = model(x)
      probability = torch.sigmoid(out_img)
      predictions = probability>0.5
      num_correct += (predictions==y).sum()
      num_pixels += torch.numel(predictions)
      loss = combined_loss(probability, y)
      total_loss += loss.item()
  accuracy = num_correct / num_pixels
  avg_loss = total_loss / len(val_loader)
  print(f'Epoch [{epoch+1}] Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
  
  return avg_loss  
  

def dice_loss(pred, target, smooth=1e-6):
    pred = pred.contiguous()
    target = target.contiguous()
    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    dice = (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    
    return 1 - dice.mean()

def combined_loss(pred, target):
    bce_loss = nn.BCEWithLogitsLoss()(pred, target)
    
    dice = dice_loss(torch.sigmoid(pred), target)
    
    return bce_loss + dice


if __name__ == '__main__':
    # Check the device we are using is GPU or CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(device)

    # # Create an UNet model object
    model = UNet().cuda()
    model_path = 'unet_model_15.pt'
    model.load_state_dict(torch.load(model_path))
    # Constants for UNet model training process
    BATCH_SIZE = 4
    NUM_EPOCHS = 15
    IMG_WIDTH = 560
    IMG_HEIGHT = 512
    # Load data
    all_data = CustomDataset('data', 'data_mask', T.Compose([T.ToTensor(), T.Resize((IMG_HEIGHT, IMG_WIDTH))]))

    # Split data into train and val
    train_indices, test_indices = train_test_split(range(len(all_data)), test_size=0.2, random_state=42)

    train_data = Subset(all_data, train_indices)
    val_data = Subset(all_data, test_indices)

    # Create loader for mini-batch gradient descent
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    # Choosing Adam as our optimizer with starting learning rate 10^-4
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # define scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    

    train(model, NUM_EPOCHS, train_loader, optimizer, scheduler, start_epoch = 10)
    # train(model, NUM_EPOCHS, train_loader, optimizer)
    