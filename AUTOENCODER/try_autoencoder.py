import torch
import torch.nn as nn
from torchvision.utils import save_image
import cv2
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import glob

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device in uso:", device)

test_path='C:/Users/Poli/Desktop/Poli project/data/test'

class CustomToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0

transform=CustomToTensor()


# Caricamento dati
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, noise_transform=None): 
        self.root_dir = root_dir
        self.transform = transform
        self.noise_transform = noise_transform
        self.file_list = glob.glob(f"{root_dir}/*.tif")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (resolution[1], resolution[0]))
        #image = np.rollaxis(image, axis=2, start=0)

        if self.transform:
            image = self.transform(image)

        if self.noise_transform:
            image = self.noise_transform(image)

        return image


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Autoencoder skype 1_______________________________________________
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        #return nn.Softmax(dim=1)(x)
        return nn.Sigmoid()(x)


test_dataset = CustomDataset(root_dir=test_path,transform=transform, noise_transform=None)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# carica il modello
model = Autoencoder()
model.load_state_dict(torch.load('models/autoencoder_08_11-56.pt'))
model.eval()
criterion = nn.MSELoss().to(device)

transform = transforms.Compose([
    transforms.ToTensor() #da img a tensore
])


resolution=(768, 1024)
losses = []
for i, test_image_path in enumerate(test_dataset.file_list):
    image = cv2.imread(test_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (resolution[1], resolution[0]))

    image_tensor = transform(image).unsqueeze(0) # serve per il batch

    with torch.no_grad():
        reconstructed_image = model(image_tensor)
    
    loss = criterion(reconstructed_image, image_tensor)
    losses.append(loss.item())

    print(f"Loss per l'immagine {test_image_path}: {loss.item()}")
    if loss.item() > 0.0005:
        print("ANOMALY DETECTED!")

        difference_tensor = torch.abs(image_tensor - reconstructed_image)


        # da tensor a array numpy
        difference_image = difference_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
        difference_image = difference_image.astype(np.uint8)
        reconstructed_image = reconstructed_image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
        reconstructed_image = reconstructed_image.astype(np.uint8)

        lab = cv2.cvtColor(difference_image, cv2.COLOR_RGB2LAB)
        a_component = lab[:,:,1]
        a_component_normalized = cv2.normalize(a_component, None, 0, 255, cv2.NORM_MINMAX)
        _, th = cv2.threshold(a_component_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blur = cv2.GaussianBlur(th, (13, 13), 11)
        heatmap_img = cv2.applyColorMap(blur, cv2.COLORMAP_JET)
        super_imposed_img = cv2.addWeighted(heatmap_img, 0.8, image, 0.2, 0)

        combined_image = np.concatenate((image, reconstructed_image, difference_image, super_imposed_img), axis=1)

        cv2.imwrite(f'output_ac/ac_combined_image_{i+1}.png', cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

average_loss = sum(losses) / len(losses)
print(f"Loss media su {len(losses)} immagini di test: {average_loss}")

