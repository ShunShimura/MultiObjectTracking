from matplotlib import pyplot as plt
import os


def combine_tracklets_and_images(tracklets, images, dir):
    # all cells
    T = len(images)
    for time in range(T):
        tracklets_upto_time = [[track[t] for t in range(time+1)] for track in tracklets]
        plt.cla()
        #　準備
        fig_all = plt.figure(figsize=(6, 5))
        ax_all = fig_all.add_subplot(111)
        ax_all.set_xlim(0, 2048), ax_all.set_ylim(0, 2048)
        ax_all.set_aspect("equal")
        cmap = plt.get_cmap("Reds")
        # image 
        ax_all.imshow(images[time], origin="lower")
        # orbit
        for track in tracklets_upto_time:
            if not isinstance(track[time], str):
                for t in range(len(track)-1):
                    if not isinstance(track[t], str) and not isinstance(track[t+1], str):
                        ax_all.plot([row[0]*20.48 for row in track[t:t+2]], [row[1]*20.48 for row in track[t:t+2]], c=cmap(0.2+0.8*t/T))
                if not isinstance(track[time], str):
                    ax_all.scatter(track[time][0]*20.48, track[time][1]*20.48, edgecolor=cmap(0.2+0.8*time/T), facecolor="none")
        dir_all_cell = dir+"/all_cells"
        os.makedirs(dir_all_cell, exist_ok=True)
        fig_all.savefig(dir_all_cell+"/time"+str(time).zfill(3)+".pdf")
        plt.close()
            
        
        