import os
from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = plt.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

folder_path = 'C:/Users/Poli/Desktop/Poli project/data/NoBag_ColorCorrectedL/'
images = load_images_from_folder(folder_path)

print('initializing SOM...')
som = MiniSom(40, 40, 3, sigma=1., learning_rate=0.2, neighborhood_function='bubble') # matrice 40x40
starting_weights = som.get_weights().copy()

# Training
j=1
for idx, img in enumerate(images, start=1):
    # Reshape e normalizzazione dei pixel dell'immagine
    pixels = np.reshape(img, (img.shape[0]*img.shape[1], 3)) / 255.
    
    #training incrementale
    print(f'training SOM on image {idx}...')
    if idx == 1:
        som.random_weights_init(pixels)
    else:
        som.train_random(pixels, 100000)

timestamp = time.strftime("%d_%H-%M")
with open(f'C:/Users/Poli/Desktop/Poli project/models/som_model_{timestamp}.pkl', 'wb') as f:
    pickle.dump(som, f)

print('displaying...')
plt.figure(figsize=(9, 6))
plt.figure(1)
plt.subplot(221)
plt.title('initial colors')
plt.imshow(starting_weights, interpolation='none')
plt.subplot(222)
plt.title('learned colors')
plt.imshow(som.get_weights(), interpolation='none')
plt.tight_layout()
plt.savefig(f'C:/Users/Poli/Desktop/Poli project/models/som_color_quantization_{timestamp}.png')
plt.show()