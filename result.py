import torch
import numpy as np
import os
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from unet_utils  import CustomDataset, UNet
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2
import shutil

def load_model(device):
    path = 'unet_model_15.pt'
    model = UNet()
    model = model.to(device)  
    model.load_state_dict(torch.load(path, map_location=device)['model_state_dict'])
    return model

def save_output_as_image(tensor, output_dir, filename):
    image = tensor.squeeze().cpu().numpy()
    image = (image * 255).astype(np.uint8)  
    pil_image = Image.fromarray(image)
    
    output_path = os.path.join(output_dir, f"{filename}")
    pil_image.save(output_path)

def evaluate_model(model, test_loader,  output_dir='output_images', device='cuda'):
    model.eval()
    total_loss = 0
    num_correct = 0
    num_pixels = 0
    
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for idx, (x, y, filename) in enumerate(tqdm(test_loader, desc="Evaluating", unit="batch")):
            x = x.to(device)
            y = y.to(device)
            
            output = model(x)
            output = torch.sigmoid(output)
            predictions = (output > 0.5).float()
            
            num_correct += (predictions == y).sum().item()
            num_pixels += torch.numel(predictions)
            
            for i in range(predictions.size(0)):
                save_output_as_image(predictions[i], output_dir, f"out_{filename[i]}")
    
    accuracy = num_correct / num_pixels
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy

def sort_nicely(file_list):
    import re
    return sorted(file_list, key=lambda x: [int(i) if i.isdigit() else i for i in re.split(r'(\d+)', x)])


def combine_images(input_folder, label_folder, model_output_folder, output_folder):
    input_images = sort_nicely([f for f in os.listdir(input_folder) if f.endswith(('.jpg'))])
    label_images = sort_nicely([f for f in os.listdir(label_folder) if f.endswith(('.jpg'))])
    model_output_images = sort_nicely([f for f in os.listdir(model_output_folder) if f.endswith(('.jpg'))])

    num_images = min(len(input_images), len(label_images), len(model_output_images))
    os.makedirs(output_folder, exist_ok=True)

    for i in range(num_images):
        input_image_path = os.path.join(input_folder, input_images[i])
        label_image_path = os.path.join(label_folder, label_images[i])
        model_output_image_path = os.path.join(model_output_folder, model_output_images[i])

        input_image = cv2.imread(input_image_path)
        label_image = cv2.imread(label_image_path)
        model_output_image = cv2.imread(model_output_image_path)

        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)
        model_output_image = cv2.cvtColor(model_output_image, cv2.COLOR_BGR2RGB)

        input_image_resized = cv2.resize(input_image, (560, 512))
        label_image_resized = cv2.resize(label_image, (560, 512))
        model_output_image_resized = cv2.resize(model_output_image, (560, 512))

        height, width, _ = input_image_resized.shape  
        offset = 10  

        cv2.putText(input_image_resized, input_images[i], (10, height - offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(label_image_resized, label_images[i], (10, height - offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(model_output_image_resized, model_output_images[i], (10, height - offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        combined_image = np.hstack((input_image_resized, label_image_resized, model_output_image_resized))

        output_filename = f"combined_{input_images[i]}"
        output_path = os.path.join(output_folder, output_filename)

        cv2.imwrite(output_path, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

def split_image(prefix):
    data_folder = 'data'
    data_mask_folder = 'data_mask'
    test_folder = 'test'
    test_mask_folder = 'test_mask'
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(test_mask_folder, exist_ok=True)
    for filename in os.listdir(data_folder):
        if filename.startswith(prefix):
            src = os.path.join(data_folder, filename)
            dst = os.path.join(test_folder, filename)
            shutil.copy(src, dst)

    for filename in os.listdir(data_mask_folder):
        if filename.startswith(prefix):
            src = os.path.join(data_mask_folder, filename)
            dst = os.path.join(test_mask_folder, filename)
            shutil.copy(src, dst)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = load_model(device=device)
    split_image('3')

    test_data = CustomDataset('test', 'test_mask', transform=T.Compose([T.ToTensor(), T.Resize((560, 512))]))
    test_loader = DataLoader(test_data, batch_size=4, shuffle=False)

    evaluate_model(model, test_loader,output_dir='output_images', device=device)

    combine_images('test', 'test_mask', 'output_images', 'output_combine')
