import tensorflow as tf
from tensorflow.keras import layers, models
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'       
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 11.8GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=11800)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)

#input_shape = (192, 256, 3)
input_shape = (384, 512, 3)
#input_shape = (768, 1024, 3)

# Load and split image paths
print("loading images")
image_paths = glob.glob("C:/Users/Poli/Desktop/Poli project/data/NoBag_ColorCorrectedL/*.tif")
print("creating dataframe")
df = pd.DataFrame({"image_path": image_paths})
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# load and preprocess
def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=input_shape[:2])
    img_array = img_to_array(img)
    img_array /= 255.0
    return img_array

# generators function
def image_generator(dataframe, batch_size):
    num_samples = len(dataframe)
    while True:
        for start in range(0, num_samples, batch_size):
            batch_paths = dataframe["image_path"][start : start + batch_size]
            batch_images = [load_and_preprocess_image(path) for path in batch_paths]
            yield np.array(batch_images), np.array(batch_images)

batch_size=4
total_images = len(train_df)
max_images_per_iteration =  len(train_df) // batch_size


print("structuring model")
encoder = models.Sequential([
        layers.InputLayer(input_shape),
        layers.Conv2D(512, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2), padding="same"),
        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2), padding="same"),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2), padding="same"),
    ]
)

decoder = models.Sequential([
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(512, (3, 3), activation="relu", padding="same"),
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(3, (3, 3), activation="softmax", padding="same"),
    ]
)

print("model compiling")
autoencoder = models.Sequential([encoder, decoder])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) 
autoencoder.compile(optimizer=optimizer, loss="mse")
autoencoder.build(input_shape=(None, *input_shape))
autoencoder.summary()

# L'idea è: il generatore di immagini è fornisce un batch di immagini alla volta durante ogni iterazione. 
# Quindi, il modello viene addestrato su quel batch per un numero specificato di epoche e poi salva i pesi. 
# Nella successiva iterazione, il generatore fornirà un nuovo batch e il modello caricherà i pesi e ripeterà il processo.
for iteration in range((total_images // max_images_per_iteration)):
    print(f"Iteration {iteration + 1}/{total_images // max_images_per_iteration}")

    if iteration > 0:
        autoencoder.load_weights(f'autoencoder_weights_iteration_{iteration - 1}.h5')
        
    train_generator = image_generator(train_df, batch_size=batch_size)
    test_generator = image_generator(test_df, batch_size=batch_size)

    autoencoder.fit(
        x=train_generator,
        steps_per_epoch=len(train_df) // batch_size,
        epochs=2,
        validation_data=test_generator,
        validation_steps=len(test_df) // batch_size
    )
    autoencoder.save_weights(f'autoencoder_weights_iteration_{iteration}.h5')

print("Addestramento completo.")



#############################################################################à
## print images
batch_images, _ = next(test_generator)

print("predicting")
reconstructed_images = autoencoder.predict(batch_images)
difference_images = np.abs(batch_images - reconstructed_images)
num_images_to_display = min(5, batch_images.shape[0])

def display_images(images, titles):
    num_images = len(images)
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)

        # Rescale the image values to the range [0, 255] for displaying with Matplotlib
        display_image = (
            (images[i] * 255).astype(np.uint8)
            if images[i].dtype != np.uint8
            else images[i]
        )

        plt.imshow(display_image)
        plt.title(titles[i])
        plt.axis("off")
    plt.show()

num_images_to_display = min(2, batch_images.shape[0])

for i in range(num_images_to_display):
    input_image = batch_images[i]
    input_title = "Input Image"

    reconstructed_image = reconstructed_images[i]
    reconstructed_title = "Reconstructed Image"

    difference_image = difference_images[i]
    difference_title = "Difference Image"

    display_images([input_image, reconstructed_image, difference_image],
        [input_title, reconstructed_title, difference_title])

print(f"Reconstructed Image - Min: {np.min(reconstructed_image)}, Max: {np.max(reconstructed_image)}")
