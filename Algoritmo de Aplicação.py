import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow import keras
from pathlib import Path

path_model = Path(r'path_to_model')
model = keras.models.load_model(path_model)
img_height = 128
img_width = 128
class_names = ['Caterpillar', 'Diabrotica speciosa', 'Healthy']

j = 0
for root, subfolder, filename in os.walk(r'path_to_images'):
    i = 0
    for file in filename:
        real_class = os.path.splitext(file)[0].split()[0]
        file_path = f'{os.path.join(root,file)}'

        img = tf.keras.utils.load_img(file_path)
        img = tf.image.resize(img, [img_height, img_width])
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
          
        ax = plt.subplot(3, 4, 2*i+1)
        plt.imshow(img.numpy().astype("uint8"))
        plt.title(real_class.capitalize(), fontsize = 8)
        plt.subplots_adjust(hspace = 0.4, wspace = 1)
        plt.axis("off")
        ax = plt.subplot(3, 4, 2*i+2)
        plt.imshow(img.numpy().astype("uint8"))
        plt.title("{} - {:.2f}%".format(class_names[np.argmax(score)], 100 * np.max(score)), fontsize = 8)
        plt.subplots_adjust(hspace = 0.4, wspace = 1)
        plt.axis("off")
        i = i+1
    fig_name = 'teste' + str(j) + '.png'
    j = j+1
    plt.savefig(fig_name, format='png')
    plt.show()