import numpy as np
from matplotlib import pyplot as plt
import glob as gl

misalignments, numbers_cell = [], []
fname = gl.glob("output/d*/evaluation.txt")
fname.sort()
for i in range(len(fname)):
    df = np.loadtxt(fname[i], delimiter=",")
    numbers_cell.append(df[0])
    misalignments.append(df[1])
    
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)
ax.plot(list(range(1, 51)), misalignments, label="misalignment", color="red")
ax.set_ylim(bottom=0)
ax.set_xlabel("threshold d on matching")
ax2 = ax.twinx()
ax2.set_ylabel("mean of misalignment between obs and pred")
ax2.plot(list(range(1, 51)), numbers_cell, label="number of cell", color="blue")
ax2.set_ylim(bottom=0)
ax.set_ylabel("number of cell")
ax.legend(loc="lower left")
ax2.legend(loc="lower right")
fig.savefig("output/evaluation.pdf")