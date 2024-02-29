import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2

som_model_path = 'C:/Users/Poli/Desktop/Poli project/models/som_model_22_02-55.pkl'
with open(som_model_path, 'rb') as f:
    som = pickle.load(f)

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = plt.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

test_folder_path = 'C:/Users/Poli/Desktop/Poli project/data/test/'
test_images = load_images_from_folder(test_folder_path)

# Quantizzazione immagini
j=0
quantization_errors = []
for img in test_images:
    #reshape + normalizzazione
    pixels = np.reshape(img, (img.shape[0]*img.shape[1], 3)) / 255.
    
    #quantizzazione dell'immagine
    qnt = som.quantization(pixels)

    #calcolo quantization error
    quantization_error=np.linalg.norm(pixels - qnt, axis=1).mean()
    print("Quantization error:", quantization_error)
    quantization_errors.append(quantization_error)
    
    #costruzione dell'immagine clusterizzata
    clustered = np.zeros(img.shape)
    for i, q in enumerate(qnt):
        clustered[np.unravel_index(i, shape=(img.shape[0], img.shape[1]))] = q
    clustered = (clustered * 255).astype(np.uint8)
    clustered = cv2.cvtColor(clustered, cv2.COLOR_RGB2BGR)
    
    #differenza tra l'immagine originale e l'immagine ricostruita
    difference = np.abs(img.astype(np.float32) - clustered.astype(np.float32))
    difference_gray = cv2.cvtColor(difference, cv2.COLOR_RGB2GRAY)
    difference_gray = np.uint8(difference_gray)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    difference_gray_rgb = cv2.cvtColor(difference_gray, cv2.COLOR_GRAY2RGB)

    lab = cv2.cvtColor(difference, cv2.COLOR_RGB2LAB)
    a_component = lab[:,:,1]
    a_component_normalized = cv2.normalize(a_component, None, 0, 255, cv2.NORM_MINMAX)
    a_component_normalized_gray = cv2.cvtColor(a_component_normalized, cv2.COLOR_GRAY2BGR)
    a_component_normalized_gray_single_channel = cv2.cvtColor(a_component_normalized_gray, cv2.COLOR_BGR2GRAY)
    a_component_normalized_gray_single_channel = a_component_normalized_gray_single_channel.astype(np.uint8)
    _, th = cv2.threshold(a_component_normalized_gray_single_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th, (13, 13), 11)
    heatmap_img = cv2.applyColorMap(blur, cv2.COLORMAP_JET)

    super_imposed_img = cv2.addWeighted(heatmap_img, 0.8, img, 0.2, 0)
    
    combined_image = np.concatenate((img, clustered, difference_gray_rgb, super_imposed_img), axis=1)
    if quantization_error>0.0024:
        print("ANOMALY DETECTED!")
        cv2.imwrite(f'C:/Users/Poli/Desktop/Poli project/output_som/som_combined_image_{j+1}.png', combined_image)
    print("done")
    j=j+1

average_qe=sum(quantization_errors) / len(quantization_errors)
print("Average quantization error: ", average_qe)
