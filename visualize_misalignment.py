import numpy as np
from matplotlib import pyplot as plt
import glob as gl

misalignments = []
fname = gl.glob("output/d*/misalignment.txt")
fname.sort()
for i in range(len(fname)):
    df = np.loadtxt(fname[i], delimiter=",")
    misalignments.append(df[1])
    
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)
ax.plot(list(range(1, 51)), misalignments)
ax.set_xlabel("threshold d")
ax.set_ylabel("mean of misalignment between obs and pred")
ax.set_ylim(bottom=0)
fig.savefig("output/misalignment.pdf")