from src import GaussProcess as gp
from src import OptimalTransport as ot
from src import KalmanFilter as kf
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN
import math
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import os

class GaussProcessTracking():
    def __init__(self, C, SamplingNumber, size=100):
        self.C = C
        self.SamplingNumber = SamplingNumber
        self.modelF_x = gp.GaussProcessModel_2d(0, 100, 101, [])
        self.modelF_y = gp.GaussProcessModel_2d(0, 100, 101, [])
        self.T = len(C)
        self.I = []
        self.dt = 1
        self.size = size
        self.IDset = set(list(range(10000000)))
        self.I.append(list(range(len(self.C[0]))))
        for i in range(len(self.C[0])):
            self.IDset.remove(min(self.IDset))
            
    def _myround1(self, val):
        return float(Decimal(str(val)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))

    def _myround0(self, val):
        return int(Decimal(str(val)).quantize(Decimal('0'), rounding=ROUND_HALF_UP))
    
    def sampling(self):
        Forces_x = self.modelF_x.sampling(self.SamplingNumber)
        Forces_y = self.modelF_y.sampling(self.SamplingNumber)
        Forces = []
        for i in range(self.SamplingNumber):
            Force = np.stack([Forces_x[i], Forces_y[i]], axis=2)
            Forces.append(Force)
        return Forces
    
    def back_predictC(self, sp, ep, sid, eid):
        prdC = []
        for (i, end_point) in zip(range(len(ep)), ep):
            end_id = eid[i]
            if end_id in sid:
                start_point = sp[sid.index(end_id)]
                prdC.append(2*end_point - start_point)
            else:
                prdC.append(end_point)
        return prdC

    def predictC_withF(self, sp, ep, sid, eid, Force, direction):
        prdC = []
        m = 1
        if direction == "forward": Force = Force
        elif direction == "backward": Force = -1 * Force
        for (i, end_point) in zip(range(len(ep)), ep):
            end_id = eid[i]
            if end_id in sid:
                start_point = sp[sid.index(end_id)]
                accelelation  = Force[int(self._myround0(start_point[0])), int(self._myround0(start_point[1]))] / m
                next_point = 2 * end_point - start_point + accelelation * self.dt**2 / 2
                prdC.append(next_point)
            else:
                prdC.append(end_point)
        return prdC


    def predictC(self, Force, Kalman=False):
        m = 1
        prdC = []
        i = len(self.I)-1
        if not Kalman:
            for id in self.I[-1]:
                if id in self.I[-2]:
                    CurrentP, PastP = self.C[i][self.I[-1].index(id)], self.C[i-1][self.I[-2].index(id)]
                    PastVelocity = CurrentP - PastP
                    NextP = CurrentP + PastVelocity * self.dt + (1/(2*m)) * Force[int(self._myround1(CurrentP[0])), int(self._myround1(CurrentP[1]))]
                    prdC.append(NextP)
                else:
                    prdC.append(self.C[i][self.I[-1].index(id)])
        else:
            # get tracks
            for id in self.I[-1]:
                track = []
                i = len(self.I)-1
                for t in range(len(self.I)):
                    if id in self.I[t]:
                        index = self.I[t].index(id)
                        track.append(self.C[t][index])               
                # kalman filter
                # obs : p_x, p_y 
                # state : p_x, p_y, v_x, v_y, a_x, a_y
                if len(track) == 1:
                    prdC.append(track[0])
                else:
                    system_matrix = [[1, 0, self.dt, 0, self.dt**2/2, 0],
                                    [0, 1, 0, self.dt, 0, self.dt**2/2],
                                    [0, 0, 1, 0, self.dt, 0],
                                    [0, 0, 0, 1, 0, self.dt],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1]
                                    ]
                    observation_matrix = [[1, 0, 0, 0, 0, 0],
                                        [0, 1, 0, 0, 0, 0] 
                                        ]
                    system_cov = [[1, 0, 0, 0, 0, 0], 
                                [0, 1, 0, 0, 0, 0], 
                                [0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 1]
                                ]
                    observation_cov = [[1, 0],
                                    [0, 1]
                                    ]
                    initial_state_mean = [track[0][0], track[0][1], (track[1][0]-track[0][0])/self.dt, (track[1][1]-track[0][1])/self.dt, 0, 0]
                    initial_state_cov = [[1, 0, 0, 0, 0, 0], 
                                        [0, 1, 0, 0, 0, 0], 
                                        [0, 0, 1, 0, 0, 0],
                                        [0, 0, 0, 1, 0, 0],
                                        [0, 0, 0, 0, 1, 0],
                                        [0, 0, 0, 0, 0, 1]
                                        ]
                    kalman = kf.KalmanFilter(system_matrix, observation_matrix, system_cov, observation_cov, initial_state_mean, initial_state_cov)
                    state_mean, state_cov = kalman.predict(1, track)
                    state_mean = state_mean.reshape(6)
                    prdC.append(state_mean[:2])
        return prdC
    
    def generateID(self, c1, c2, d=10):
        N, M = len(c1), len(c2)
        C, X = np.zeros((N+M, N+M)), np.zeros((N+M, N+M))
        for i in range(N+M):
            for j in range(N+M):
                if i < N and j < M:
                    C[i, j] = math.dist(c1[i], c2[j])
                else:
                    C[i, j] = d
        X, misalignment, number_match = ot.OptimalTransport(C, X)
        X = X[:N, :M]
        NewID = []
        for j in range(M):
            if 1 in [row[j] for row in X]:
                i = [row[j] for row in X].index(1)
                id = self.I[-1][i]
                NewID.append(id)
            else:
                NewID.append(min(self.IDset))
                self.IDset.remove(min(self.IDset))
        return NewID, misalignment, number_match
    
    def back_generateID(self, c1, c2, updated_t):
        d=.2*self.size
        N, M = len(c1), len(c2)
        C, X = np.zeros((N+M, N+M)), np.zeros((N+M, N+M))
        for i in range(N+M):
            for j in range(N+M):
                if i < N and j < M:
                    C[i, j] = math.dist(c1[i], c2[j])
                else:
                    C[i, j] = d
        X, score = ot.OptimalTransport(C, X)
        X = X[:N, :M]
        NewID = []
        for j in range(M):
            if 1 in [row[j] for row in X]:
                index = [row[j] for row in X].index(1)
                NewID.append(self.I[updated_t+1][index])
            else:
                NewID.append(min(self.IDset))
                self.IDset.remove(min(self.IDset))
        return NewID, score

    def for_generateID(self, c1, c2, s_t):
        d=.2*self.size
        N, M = len(c1), len(c2)
        C, X = np.zeros((N+M, N+M)), np.zeros((N+M, N+M))
        for i in range(N+M):
            for j in range(N+M):
                if i < N and j < M:
                    C[i, j] = math.dist(c1[i], c2[j])
                else:
                    C[i, j] = d
        X, score = ot.OptimalTransport(C, X)
        X = X[:N, :M]
        NewID = []
        for j in range(M):
            if 1 in [row[j] for row in X]:
                index = [row[j] for row in X].index(1)
                NewID.append(self.I[s_t][index])
            else:
                NewID.append(min(self.IDset))
                self.IDset.remove(min(self.IDset))
        return NewID, score
    
    def generateNewID(self, length):
        NewID = []
        for i in range(length):
            NewID.append(min(self.IDset))
            self.IDset.remove(min(self.IDset))
        return NewID

    def evaluate(self, id):
        #未開発
        return 0
    
    def updateGP(self):
        m = 1
        # get current tracks
        tracks = []
        for id in self.I[-1]:
            track = []
            i = len(self.I)-1
            for t in range(len(self.I)):
                if id in self.I[t]:
                    index = self.I[t].index(id)
                    track.append(self.C[t][index])
            tracks.append(track)
        NewData_x, NewData_y = [], []
        # calculate (x, F)
        for track in tracks:
            if len(track) > 2:
                x = track[-3]
                F = m * (track[-1]-2*track[-2]+track[-3])/self.dt**2
                NewData_x.append([x, F[0]])
                NewData_y.append([x, F[1]])
            else:
                continue
        # update GP
        self.modelF_x.update(NewData_x)
        self.modelF_y.update(NewData_y)
        return 0
    
    def updateGP_F(self, F):
        # get current tracks
        tracks = []
        for id in self.I[-1]:
            track = []
            i = len(self.I)-1
            for t in range(len(self.I)):
                if id in self.I[t]:
                    index = self.I[t].index(id)
                    track.append(self.C[t][index])
            tracks.append(track)
        NewData_x, NewData_y = [], []
        # calculate (x, F)
        for track in tracks:
            if len(track) > 2:
                x = track[-3]
                right_F = F[self._myround0(x[0]), self._myround0(x[1])]
                NewData_x.append([x, right_F[0]])
                NewData_y.append([x, right_F[1]])
            else:
                continue
        # update GP
        self.modelF_x.update(NewData_x)
        self.modelF_y.update(NewData_y)
        
        return 0
    
    def visualize(self, dir, file_name):
        # tracklets 生成
        treated_id, tracklets = [], []
        for t in range(self.T):
            for id in self.I[t]:
                if not id in treated_id:
                    treated_id.append(id)
                    tracklet = []
                    tracklet.extend(["nan"]*t)
                    for t2 in range(t, self.T):
                        if id in self.I[t2]:
                            index = self.I[t2].index(id)
                            tracklet.append(self.C[t2][index].tolist())
                        else:
                            tracklet.extend(["nan"]*(self.T-t2))
                            break
                    tracklets.append(tracklet)
                    
        # visualize
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        ax.set_xlim(0, self.size), ax.set_ylim(0, self.size)
        ax.set_aspect("equal")
        cmap = plt.get_cmap("Blues")
        # orbit 生成
        for tracklet in tracklets:
            for t in range(len(tracklet)-1):
                if not isinstance(tracklet[t], str) and not isinstance(tracklet[t+1], str):
                    ax.plot([row[0] for row in tracklet][t:t+2], [row[1] for row in tracklet][t:t+2], c=cmap(0.2+0.8*t/self.T))

        # split point 生成
        for tracklet in tracklets:
            for t in range(0, len(tracklet)):
                if not isinstance(tracklet[t], str):
                    ax.scatter(tracklet[t][0], tracklet[t][1], color=cmap(0.2+0.8*t/self.T), s=5)
                    
        # colorbar 生成            
        c = [t for t in range(10)]
        x, y = [-1 for i in range(len(c))], [-1 for i in range(len(c))]
        sc = ax.scatter(x, y, c=c, cmap="Blues")
        cb = fig.colorbar(sc, aspect=40, pad=0.12, ticks=[0, 4.5, 9], orientation='vertical')
        cb.ax.set_yticklabels(["early", "middle", "later"])
        cb.ax.tick_params(labelsize=15)
        
        fig.savefig(dir+"/"+file_name)
        
        # each cell
        i = 1
        for tracklet in tracklets:
            plt.cla()
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111)
            ax.set_xlim(0, self.size), ax.set_ylim(0, self.size)
            ax.set_aspect("equal")
            cmap = plt.get_cmap("Blues")
            for t in range(len(tracklet)-1):
                if not isinstance(tracklet[t], str) and not isinstance(tracklet[t+1], str):
                    ax.plot([row[0] for row in tracklet][t:t+2], [row[1] for row in tracklet][t:t+2], c=cmap(0.2+0.8*t/self.T))
            for t in range(0, len(tracklet)):
                if not isinstance(tracklet[t], str):
                    ax.scatter(tracklet[t][0], tracklet[t][1], color=cmap(0.2+0.8*t/self.T), s=5)
            c = [t for t in range(10)]
            x, y = [-1 for i in range(len(c))], [-1 for i in range(len(c))]
            sc = ax.scatter(x, y, c=c, cmap="Blues")
            cb = fig.colorbar(sc, aspect=40, pad=0.12, ticks=[0, 4.5, 9], orientation='vertical')
            cb.ax.set_yticklabels(["early", "middle", "later"])
            cb.ax.tick_params(labelsize=15)
            os.makedirs(dir+"/each_cell", exist_ok=True)
            fig.savefig(dir+"/each_cell/"+str(i).zfill(3)+".pdf")
            plt.close()
            i += 1

        

        return 0
    
    def RegressionGPmodlVisualize(self, fname):
        mean_x = self.modelF_x.PosteriorProbabilityMean()
        mean_y = self.modelF_y.PosteriorProbabilityMean()
        mean = np.stack([mean_x, mean_y], axis=2)
        absolute = np.array([[(mean[i, j, 0]**2+mean[i, j, 1]**2)**0.5 for j in range(len(mean[0]))] for i in range(len(mean))])
        angle = np.array([[math.atan2(mean[i, j, 1], mean[i, j, 0])+math.pi for j in range(len(mean))] for i in range(len(mean))])
        factor = 1/absolute.max()
        factor = .2
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        ax.set_xlim(25, 75)
        ax.set_ylim(25, 75)
        pixel = len(mean)
        x, y = [j for j in range(pixel) for i in range(pixel)], [i for j in range(pixel) for i in range(pixel)]
        mean_rs = mean.reshape(pixel**2, 2)
        angle_rs = angle.reshape(pixel**2)
        u, v = mean_rs[:, 0]*factor, mean_rs[:, 1]*factor
        cmap = plt.get_cmap("hsv")
        colors = [cmap(value/(2*math.pi)) for value in angle_rs]
        ax.quiver(x, y, u, v, angles="uv", scale_units="xy", scale=1, color=colors)
        
        # color bar
        c = [t for t in range(10)]
        x, y = [-1 for i in range(len(c))], [-1 for i in range(len(c))]
        sc = ax.scatter(x, y, c=c, cmap="hsv")
        cb = fig.colorbar(sc, aspect=40, pad=0.12, ticks=[0, 4.5, 9], orientation='vertical')
        cb.ax.set_yticklabels(["-π", "0", "π"])
        cb.ax.set_xlabel("angle", size=15, labelpad=20)
        cb.ax.tick_params(labelsize=15)
        fig.savefig(fname)
        return factor
    
    def InputToGPmodlVisualize(self, fname, factor):
        data_x = self.modelF_x.data
        data_y = self.modelF_y.data
        if len(data_x) != len(data_y):
            print("data_x and data_y must have the same length!")
        x, y, u, v = [data[0][0] for data in data_x], [data[0][1] for data in data_x], [data[1] for data in data_x], [data[1] for data in data_y]
        absolute = np.array([(s**2+t**2)**0.5 for (s, t) in zip(u, v)])
        angle = [math.atan2(t, s)+math.pi for (s, t) in zip(u, v)]
        u = np.array(u) * factor
        v = np.array(v) * factor
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        ax.set_xlim(25, 75)
        ax.set_ylim(25, 75)
        cmap = plt.get_cmap("hsv")
        colors = [cmap(value/(2*math.pi)) for value in angle]
        ax.quiver(x, y, u, v, angles="uv", scale_units="xy", scale=1, color=colors)
        
        # color bar
        c = [t for t in range(10)]
        x, y = [-1 for i in range(len(c))], [-1 for i in range(len(c))]
        sc = ax.scatter(x, y, c=c, cmap="hsv")
        cb = fig.colorbar(sc, aspect=40, pad=0.12, ticks=[0, 4.5, 9], orientation='vertical')
        cb.ax.set_yticklabels(["-π", "0", "π"])
        cb.ax.set_xlabel("angle", size=15, labelpad=20)
        cb.ax.tick_params(labelsize=15)
        fig.savefig(fname)
        return 0
    
    def get_tracklets(self):
        treated_id, tracklets = [], []
        for t in range(self.T):
            for id in self.I[t]:
                if not id in treated_id:
                    treated_id.append(id)
                    tracklet = []
                    tracklet.extend(["nan"]*t)
                    for t2 in range(t, self.T):
                        if id in self.I[t2]:
                            index = self.I[t2].index(id)
                            tracklet.append(self.C[t2][index].tolist())
                        else:
                            tracklet.extend(["nan"]*(self.T-t2))
                            break
                    tracklets.append(tracklet)
        return tracklets        