import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tkinter.ttk as ttk
import os

from tensorflow import keras
from pathlib import Path


python_path = os.getcwd()
model_path = python_path + r'\Final Model.keras'
model = keras.models.load_model(model_path)
img_height = 128
img_width = 128


def image_load():
  global img_array, filepath, img
  # Opens a window to select the image
  filepath = filedialog.askopenfilename()
  if filepath:
    # Loads the image using PIL
    img = Image.open(filepath)
    img_resize = img.resize((img_height, img_width))
    img_tk = ImageTk.PhotoImage(img)
    # Displays the image on the label
    label_imagem.config(image=img_tk)
    label_imagem.image = img_tk
    # Converts the image to a NumPy array
    img_array = tf.keras.utils.img_to_array(img_resize)
    img_array = tf.expand_dims(img_array, 0)


def image_classify():
  global predictions, score, class_names

  class_names = ['Caterpillar', 'Diabrotica speciosa', 'Healthy']

  if img_array is not None:
    # Performs prediction using the model
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    # Displays the result on the label
    label_classification.config(text=class_names[np.argmax(score)])
    label_accuracy.config(text="{:.2f}%".format(100 * np.max(score)))


def save_image():
  global savepath
  plt.imshow(img)
  plt.title("{} - {:.2f}%".format(class_names[np.argmax(score)], 100 * np.max(score)), fontsize=12, pad=10, fontweight='bold')
  plt.axis("off")

  filename = os.path.basename(filepath)
  dirname = os.path.dirname(filepath)
  name, ext = os.path.splitext(filename)
  savename = name + " Classified" + ext
  savepath = os.path.join(dirname, savename)
  plt.savefig(savepath)

# Creates the main window
window = tk.Tk()

color = "#DCDFE0"

width_screen = window.winfo_screenwidth()
height_screen = window.winfo_screenheight()

width_window = int(width_screen * 0.6)
height_window = int(height_screen * 0.6)

window.geometry(f"{width_window}x{height_window}")
window.configure(bg=color)
window.title("Image Classifier")  # Changed title to English

style = ttk.Style()
style.theme_use('clam')
style.configure('TButton', font=('Arial', 12), padding=10)
style.configure('Load.TButton', background='#008CBA', foreground='white')
style.configure('Classify.TButton', background='#4CAF50', foreground='white')
style.configure('Save.TButton', background='#FFA500', foreground='white')

# Configures the grid of the main window
window.grid_rowconfigure(0, weight=0, minsize=int(0.1 * height_window))  # Allows the row to expand
window.grid_rowconfigure(1, weight=1)
window.grid_columnconfigure(0, weight=1)  # Left column (40%)
window.grid_columnconfigure(1, weight=1)  # Right column (60%)

# Creates frames to divide the screen
left_frame = tk.Frame(window, bg=color)
left_frame.grid(row=1, column=0, sticky="nsew")  # Positions in the left column

left_right = tk.Frame(window, bg=color)
left_right.grid(row=1, column=1, sticky="nsew")

# Text for the work in the left frame, centered vertically
title = tk.Label(window, text="Trabalho de Conclusão de Curso: Técnicas de Machine Learning Aplicadas à Detecção e Classificação de Pragas na Cultura de Soja",
                  font=("Arial", 14), justify="center", anchor="center", bg='#4CAF50', pady=20)
title.grid(row=0, column=0, sticky="nsew", columnspan=2)  # Centers using grid

# Configures the left frame to resize with the window
left_frame.rowconfigure(0, weight=1)
left_frame.rowconfigure(1, weight=0, minsize=510)
left_frame.rowconfigure(2, weight=1)

left_frame.columnconfigure(0, weight=1, minsize=520)

left_right.rowconfigure(0, weight=0, minsize=int(0.3 * height_window))
left_right.rowconfigure(1, weight=1)
left_right.rowconfigure(2, weight=1)
left_right.rowconfigure(3, weight=1)
left_right.rowconfigure(4, weight=1)
left_right.rowconfigure(5, weight=1)
left_right.rowconfigure(6, weight=1)
left_right.rowconfigure(7, weight=0, minsize=int(0.3 * height_window))

left_right.columnconfigure(0, weight=1)

# Buttons in the right frame
button_Load = ttk.Button(left_frame, text="Carregar Imagem", command=image_load, style='Load.TButton')
button_Load.grid(row=0, column=0)

# Blank space to display the image with border
label_imagem = tk.Label(left_frame, bg=color)
label_imagem.grid(row=1, column=0)

# Label "Image for classification" in the right frame
label_imagem_text = tk.Label(left_frame, text="Imagem para classificação", font=("Arial", 12), bg=color)
label_imagem_text.grid(row=2, column=0)

# Label to display the classification result
button_Classify = ttk.Button(left_right, text="Classificar Imagem", command=image_classify, style='Classify.TButton')
button_Classify.grid(row=1, column=0)

label_classification_text = tk.Label(left_right, text="Classificação:", font=("Arial", 12), bg=color)
label_classification_text.grid(row=2, column=0)

label_classification = tk.Label(left_right, text="", font=("Roboto", 12, "bold"), fg="#333333", bg=color)
label_classification.grid(row=3, column=0)

# Label to display the confidence percentage
label_accuracy_text = tk.Label(left_right, text="Confiança:", font=("Arial", 12), bg=color)
label_accuracy_text.grid(row=4, column=0)

label_accuracy = tk.Label(left_right, text="", font=("Roboto", 12, "bold"), fg="#333333", bg=color)
label_accuracy.grid(row=5, column=0)

save_button = ttk.Button(left_right, text="Salvar Classificação", command=save_image, style='Save.TButton')
save_button.grid(row=6, column=0)

# Starts the main loop of the window
window.mainloop()