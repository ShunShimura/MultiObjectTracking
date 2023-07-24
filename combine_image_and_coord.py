from matplotlib import pyplot as plt
import glob as gl
from PIL import Image
import os
import numpy as np

folder_image = "image_data"
folder_coord = "real_data"
fname_image = gl.glob(folder_image+"/*")
fname_coord = gl.glob(folder_coord+"/*")
fname_coord.sort()
fname_image.sort()

dir_output = "fig/coord_image"
os.makedirs(dir_output, exist_ok=True)

for t in range(len(fname_coord)):
    plt.cla()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.imshow(Image.open(fname_image[t]), origin="lower")
    coord = np.loadtxt(fname_coord[t], delimiter=",")
    if coord.ndim == 1:
        coord = np.array([coord])
    ax.scatter(coord[:, 0]*20.48, coord[:, 1]*20.48, edgecolor="red", facecolor="none")
    fig.savefig(dir_output+"/time"+str(t).zfill(3)+".pdf")
    plt.close()
        
