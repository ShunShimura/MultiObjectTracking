import numpy as np
from matplotlib import pyplot as plt
import glob as gl

misalignments = []
fname = gl.glob("output/d*/misalignment.txt")
fname.sort()
for i in range(len(fname)):
    df = np.loadtxt(fname[i], delimiter=",")
    misalignments.append(df[1])
    
print(misalignments)