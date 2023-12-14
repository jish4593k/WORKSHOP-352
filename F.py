# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 17:43:40 2020

@author: Sounmay Mishra
@contributor: Sandipta Sahu
"""

import os
import cv2
import torch
import pickle
from pathlib import Path
from torchvision import transforms
from tkinter import Tk, Label, Button, filedialog

# Constants for file paths
DATA_DIR = Path(os.getcwd()) / 'clean_data'
IM_DIR = Path(os.getcwd()) / 'images'

# PyTorch transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((200, 200)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

def preprocessing(image):
    return transform(image)

data = []
label = []

# Iterate through images in the 'images' directory
for i in IM_DIR.iterdir():
    image = cv2.imread(str(i))
    image_tensor = preprocessing(image)
    data.append(image_tensor)
    label.append(i.stem.split('_')[0])

data = torch.stack(data)
label = torch.tensor(label)

# Save image data and labels using torch
torch.save(data, DATA_DIR / 'images.pt')
torch.save(label, DATA_DIR / 'labels.pt')

# Tkinter GUI setup for saving the preprocessed data
def save_preprocessed_data():
    global data, label

    folder_path = filedialog.askdirectory(title="Select Folder to Save Preprocessed Data")

    if folder_path:
        torch.save(data, os.path.join(folder_path, 'images.pt'))
        torch.save(label, os.path.join(folder_path, 'labels.pt'))

# Tkinter GUI to trigger the saving of preprocessed data
root = Tk()
root.title("Save Preprocessed Data")

btn_save = Button(root, text="Save Preprocessed Data", command=save_preprocessed_data)
btn_save.pack(pady=10)

root.mainloop()
