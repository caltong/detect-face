import os
import numpy as np
from PIL import Image

data_folder_path = os.path.join('data', 'data_covered')
files = os.listdir(data_folder_path)
x_train = []
for file in files:
    path = os.path.join(data_folder_path, file)
    image = Image.open(path)
    x_train.append(image)

y_train = []
with open(os.path.join('data', 'margin.txt')) as file:
    lines = file.readlines()
    for line in lines:
        y_train.append(line)

y_train = np.array(y_train)
