import Tracking as gpt
import numpy as np
import glob as gl
import os
import warnings
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import sys

def main1(C, SamplingNumber, size): # 位置マッチング
    gpt1 = gpt.GaussProcessTracking(C, 10)
    ID, _ = gpt1.generateID((C[0]), C[1])
    gpt1.I.append(ID)
    for t in range(1, gpt1.T-1):
        print("\n-----------------------------------")
        print("t = "+str(t).zfill(3))
        if not isinstance(C[t], str) and not isinstance(C[t+1], str):
            ID, _ = gpt1.generateID(C[t], C[t+1])
        elif isinstance(C[t], str) and not isinstance(C[t+1], str):
            ID = gpt1.generateNewID(len(C[t+1]))
        elif isinstance(C[t+1], str):
            ID = []
        gpt1.I.append(ID)
    return gpt1

def main2(C, SamplingNumber, size, d=5): # 予測位置マッチング
    predicts, detects = [], []
    for c in C:
        if not isinstance(c, str): detects.append(c[0])
    gpt1 = gpt.GaussProcessTracking(C, 10)
    ID, _, _ = gpt1.generateID(C[0], C[1])
    gpt1.I.append(ID)
    sum_misalignment, sum_number_match = 0, 0
    for t in range(1, gpt1.T-1):
#        print("\n-----------------------------------")
#        print("t = "+str(t).zfill(3))
        if not isinstance(C[t], str) and not isinstance(C[t+1], str):
            prdC = gpt1.predictC(np.zeros((size+1, size+1)))
            predicts.append(prdC[0])
            ID, misalignment, number_match = gpt1.generateID(prdC, C[t+1], d=d)
            sum_misalignment += misalignment
            sum_number_match += number_match
        elif isinstance(C[t], str) and not isinstance(C[t+1], str):
            ID = gpt1.generateNewID(len(C[t+1]))
        elif isinstance(C[t+1], str):
            ID = []
        gpt1.I.append(ID)
    return gpt1, sum_misalignment/sum_number_match

def main3(C, SamplingNumber, size): # 予測位置×力学予測マッチング
    #print("めっちゃ時間かかるよーーーー")
    gpt1 = gpt.GaussProcessTracking(C, 5)
    ID = gpt1.generateID(C[0], C[1])
    gpt1.I.append(ID)
    for t in range(1, gpt1.T-1):
#        print("\n-----------------------------------")
#        print("t = "+str(t).zfill(3))
        if not isinstance(C[t], str) and not isinstance(C[t+1], str):
            Forces = gpt1.sampling()
            MaxScore = 0
            for Force in Forces:
                prdC = gp1.predictC(Force)
                ID, Score = gpt1.generateID(prdC, C[t])
                if Score > MaxScore:
                    MaxScore, BestF, BestID = Score, Force, ID
            gpt1.I.append(BestID)
            gpt1.updateGP()
        elif isinstance(C[t], str) and not isinstance(C[t+1], str):
            ID = gpt1.generateNewID(len(C[t+1]))
        elif isinstance(C[t+1], str):
            ID = []
    return gpt1

#########################################################################################################

d = int(sys.argv[1])
print("-"*50)
print("d = "+str(d).zfill(2))

# data load
data_folder = "real_data"
file_name = gl.glob(data_folder+"/*")
file_name.sort()
C = []
for FileName in file_name:
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try:
            df = np.loadtxt(FileName, delimiter=",")
            if df.ndim == 1:
                df = np.array([df])
            C.append(df)
        except UserWarning:
            C.append("nan")
        except IOError:
            C.append("nan")
        
SamplingNumber = 1
size = 100

"""
#number = int(input("Please write number 1 to 7 : "))
number = 2

if number == 1:
    model = main1(C, SamplingNumber, size)
elif number == 2:
    model = main2(C, SamplingNumber, size)
elif number == 3:
    model = main4(C, SamplingNumber, size)
else:
    print("number should be 1 to 3 !")

# data output 
output_folder = "output/identification"
os.makedirs(output_folder, exist_ok=True)
for t in range(len(model.I)):
    np.savetxt(output_folder+"/time"+str(t).zfill(3)+".txt", model.I[t], delimiter=",", fmt="%d")
"""

print("tracking...")
model, misalignment = main2(C, SamplingNumber, size, d=d)

# ouput 
print("outputting...")
output_folder_ID = "output/d"+str(d).zfill(2)+"/identification"
os.makedirs(output_folder_ID, exist_ok=True)
for t in range(len(model.I)):
    np.savetxt(output_folder_ID+"/time"+str(t).zfill(3)+".txt", model.I[t], delimiter=",", fmt="%d")
number_cell = max(model.I[-1])
np.savetxt("output/d"+str(d).zfill(2)+"/evaluation.txt", [number_cell, misalignment], delimiter=",", fmt="%d")