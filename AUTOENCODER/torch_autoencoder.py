import torch
from torchsummary import summary
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.utils import save_image
import glob
from tqdm import tqdm
import time
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

dataset_path = 'C:/Users/Poli/Desktop/Poli project/data/NoBag_ColorCorrectedL'
test_path = 'C:/Users/Poli/Desktop/Poli project/data/test'

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device in uso:", device)

class CustomToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0
        
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

# noise
class AddNoise(object):
    def __init__(self, noise_factor=0.2):
        self.noise_factor = noise_factor

    def __call__(self, img):
        return img + self.noise_factor * torch.randn_like(img)

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

# Config
resolution=(768, 1024)
#resolution=(384, 512)
#dropout_prob = 0.01
batch_size = 16
learning_rate = 0.0001
num_epochs = 30
noise_value = 0

# Preprocess and tranformation in tensors
transform=CustomToTensor()

#instantiations
noise_transform = AddNoise(noise_factor=noise_value)
custom_dataset = CustomDataset(root_dir=dataset_path, transform=transform, noise_transform=noise_transform)
test_dataset = CustomDataset(root_dir=test_path,transform=transform, noise_transform=noise_transform)
train_dataset, val_dataset = train_test_split(custom_dataset, test_size=0.2, random_state=42) 
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print("Loading and preprocessing data: done.")

autoencoder = Autoencoder()
criterion = nn.MSELoss().to(device)
#optimizer = optim.Adadelta(autoencoder.parameters(), lr=learning_rate)
#optimizer = optim.RMSprop(autoencoder.parameters(), lr=learning_rate, alpha=0.9) #come adam
#optimizer= optim.SGD(autoencoder.parameters(), lr=learning_rate, momentum=0.9)
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
autoencoder.to(device)
summary(autoencoder, input_size=(3, resolution[0], resolution[1]), device=device)
print("Model Initialization: done.")


timestamp = time.strftime("%d_%H-%M")


# Training e Validation
for epoch in range(num_epochs):
    # Training
    autoencoder.train()
    train_loss = 0.0
    for data in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs} - Training', unit='batch'):
        data = data.to(device)
        optimizer.zero_grad()
        reconstructions = autoencoder(data)
        loss = criterion(reconstructions, data)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_dataloader)

    # Validation
    autoencoder.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in tqdm(val_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs} - Validation', unit='batch'):
            data = data.to(device)
            reconstructions = autoencoder(data)
            loss = criterion(reconstructions, data)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_dataloader)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

# Test e salvataggio immagini
autoencoder.eval()
test_loss = 0.0
with torch.no_grad():
    for data in tqdm(test_dataloader, desc=f'Testing', unit='batch'):
        data = data.to(device)
        model = autoencoder.to(device)
        reconstructions = autoencoder(data)
        loss = criterion(reconstructions, data)
        test_loss += loss.item()

        # Salva le immagini di test, ricostruzioni e differenze
        if epoch == num_epochs - 1:
            test_data_cpu = data.cpu()
            reconstructions_cpu = reconstructions.cpu()
            difference = torch.abs(test_data_cpu - reconstructions_cpu)

            all_together = torch.cat([test_data_cpu, reconstructions_cpu, difference], dim=0)
            save_image(all_together, f'pics/torch/test_results_{timestamp}.png', nrow=batch_size)

avg_test_loss = test_loss / len(test_dataloader)
print(f'Test Loss: {avg_test_loss:.4f}')

print("Model training and testing: done.")

torch.save(autoencoder.state_dict(), f'models/autoencoder_{timestamp}.pt')
print("Model saving: done.")



    
    