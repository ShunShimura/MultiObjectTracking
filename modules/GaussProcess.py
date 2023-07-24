import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN

def my_round(val, digit=0):#digit=1で少数第一位まで返す
    p = 10 ** digit
    return (val * p * 2 + 1) // 2 / p

def my_round0(val):
    return int(Decimal(str(val)).quantize(Decimal('0'), rounding=ROUND_HALF_UP))

def my_round1(val):
    return float(Decimal(str(val)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))

import time
def time_display(s_time):
    e_time = time.time()
    print("Runtime = "+str(e_time-s_time)+"\n")
    return time.time()

class GaussProcessModel_2d():
    def __init__(self, lb, ub, size, data, theta1=1, theta2=20, theta3=0):
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3
        self.lb = lb
        self.ub = ub
        self.size = size                
        self.step = my_round0((self.ub-self.lb)/(self.size-1))
        self.kernel_map = dict()
        for i in range(size):
            x1 = my_round1(lb + i * self.step)
            for j in range(size):
                x2 = my_round1(lb + j * self.step)
                self.kernel_map[(x1, x2)] = self.theta1 * math.e**(-(x1**2+x2**2)**0.5/self.theta2)
        self.data = data
        if len(self.data) != 0:
            self.x_obs = self._get_x_obs()
            self.n = len(self.x_obs)
            self.map_obs = self._get_map_obs()
            self.y_obs = self._get_y_obs()
        self.x_prd = self._get_x_prd()
        self.map_prd = self._get_map_prd()
        self.m = len(self.x_prd)
        if len(self.data) != 0:
            self.K = self._get_K()
            self.k_star = self._get_k_star()
        self.k_2star = self._get_k_2star()

    def _kernel(self, x1, x2):
        dx, dy= my_round0(abs(x1[0]-x2[0])), my_round0(abs(x1[1]-x2[1]))
        return self.kernel_map[(dx, dy)]
    
    def _kernel_xm(self, x1, m2):
        """
        値と配列のカーネル : 1次元配列になる
        """
        kernel = np.zeros(len(m2))
        for j in range(len(m2)):
            kernel[j] = self._kernel(x1, m2[j])
        return kernel
            
    def _kernel_mm(self, m1, m2):
        """
        配列と配列のカーネル : 2次元配列になる
        """
        if len(m1) != len(m2):
            print("two lengths must be the same.")
        kernel = np.zeros(len(m1))
        for i in range(len(m1)):
            for j in range(len(m2)):
                kernel[i, j] = self._kernel(m1[i], m2[j])
        return kernel
    
    def _get_x_obs(self):
        if len(self.data) == 0:
            return []
        else:
            observes = [row[0] for row in self.data]
            return [[my_round0(row[0]), my_round0(row[1])] for row in observes]

    def _get_y_obs(self):
        if len(self.data) == 0:
            return []
        else:
            return [row[1] for row in self.data]
    
    def _get_x_prd(self):
        grid_all_x = [self.lb+i*(self.ub-self.lb)/(self.size-1) for i in range(self.size)]
        if len(self.data) == 0:
            return [[grid_all_x[i], grid_all_x[j]] for i in range(self.size) for j in range(self.size)]
        else:
            return [[grid_all_x[i], grid_all_x[j]] for i in range(self.size) for j in range(self.size) if not [grid_all_x[i], grid_all_x[j]] in self.x_obs]
        
    def _get_map_obs(self):
        dictionary = dict()
        for i in range(self.n):
            dictionary[tuple(self.x_obs[i])] = i
        return dictionary
        
    def _get_map_prd(self):
        dictionary = dict()
        for i in range(len(self.x_prd)):
            dictionary[tuple(self.x_prd[i])] = i
        return dictionary
    
    def _update_map_prd(self, x, index_erase):
        self.map_prd.pop(tuple(x))
        for key in self.map_prd:
            if self.map_prd[key] > index_erase:
                self.map_prd[key] -= 1
        return 0

    def _get_K(self):
        if len(self.data) == 0:
            return np.array([])
        else:
            K = [[self._kernel(self.x_obs[i], self.x_obs[j]) for j in range(self.n)] for i in range(self.n)]
            return np.array(K)

    def _get_k_star(self):
        k_star = [[self._kernel(self.x_obs[i], self.x_prd[j]) for j in range(self.m)] for i in range(self.n)]
        return np.array(k_star)

    def _get_k_2star(self):
        k_2star = np.zeros((self.m,self.m))
        for i in range(self.m):
            for j in range(self.m):
                k_2star[i][j] = self._kernel(self.x_prd[i], self.x_prd[j])
                k_2star[j][i] = deepcopy(k_2star[i][j])
                if i > j:
                    break
        return k_2star

    def sampling(self, sampling_number=1):
        # dataあり
        if len(self.data) != 0:
            k_star_T = self.k_star.T
            K_inv = np.linalg.inv(self.K)
            mu = k_star_T @ K_inv @ self.y_obs
            cov = self.k_2star - k_star_T @ K_inv @ self.k_star
            decomposition_matrix, decomposition = get_decomposition_matrix(cov)
            if sampling_number == 1:
                return [self.sample_multivariate_normal(mu, decomposition_matrix, decomposition)]
            else:
                samplings = []
                for i in range(sampling_number):
                    sampling = self.sample_multivariate_normal(mu, decomposition_matrix, decomposition)
                    samplings.append(sampling)
                return samplings
        # data なし
        else:
            mu = np.zeros(self.size**2)
            cov = self.k_2star
            decomposition_matrix, decomposition = get_decomposition_matrix(cov)
            if sampling_number == 1:
                return [self.sample_multivariate_normal(mu, decomposition_matrix, decomposition)]
            else:
                samplings = []
                for i in range(sampling_number):
                    sampling = self.sample_multivariate_normal(mu, decomposition_matrix, decomposition)
                    samplings.append(sampling)
                return samplings
            
    def PosteriorProbabilityMean(self):
        k_star_T = self.k_star.T
        K_inv = np.linalg.inv(self.K)
        mu = k_star_T @ K_inv @ self.y_obs
        mean = np.zeros((self.size, self.size))
        #ここからの処理時間長い
        s_time = time.time()
        for i in range(self.size):
            x = my_round0(self.lb + i * self.step)
            for j in range(self.size):
                y = my_round0(self.lb + j * self.step)
                if [x, y] in self.x_obs:
                    index = self.map_obs[(x, y)]
                    mean[i, j] = self.data[index][1]
                else:
                    index = self.x_prd.index([x, y])
                    index = self.map_prd[(x, y)]
                    mean[i, j] = mu[index]
        time_display(s_time)
        return mean

    def get_decomposition_matrix(self, cov: np.array) -> (tuple, str):
        try:
            return np.linalg.cholesky(cov), "cholesky"
        except np.linalg.LinAlgError as e:
            return np.linalg.svd(cov), "SVD"

    def sample_multivariate_normal(self, mean: np.array, decomposition_matrix: tuple,
                                decomposition: str) -> np.array:
        if decomposition == "cholesky":
            standard_normal_vector = np.random.standard_normal(len(decomposition_matrix))
            return decomposition_matrix @ standard_normal_vector + mean
        elif decomposition == "SVD":
            u, s, vh = decomposition_matrix
            standard_normal_vector = np.random.standard_normal(len(u))
            return u @ np.diag(np.sqrt(s)) @ vh @ standard_normal_vector + mean
    
    def update(self, newdatas): # 処理長め
        if len(self.data) != 0:
            for newdata in newdatas:
                x, y = [my_round0(newdata[0][0]), my_round0(newdata[0][1])], newdata[1]
                # x_obs と x_prd の更新 ※indexを取得しておく
                if not x in self.x_obs:    #未観測のデータ点
                    #追加群
                    self.data.append([x, y])
                    self.x_obs.append(x)
                    self.n += 1
                    self.map_obs[tuple(x)] = self.n-1
                    #削除群
                    index = self.map_prd[tuple(x)]
                    self.x_prd.remove(x)
                    self.m -= 1
                    self._update_map_prd(x, index)
                    s_time = time.time()
                    # k_star k_2star の更新
                    #print("追加カーネル更新")
                    self.k_star = np.delete(self.k_star, index, 1)
                    self.k_2star = np.delete(np.delete(self.k_2star, index, 0), index, 1)
                    #s_time = time_display(s_time)
                    # k_star K の更新
                    self.k_star = np.vstack([self.k_star, self._kernel_xm(x, np.array(self.x_prd)).reshape(1, self.m)])
                    self.K = np.hstack([self.K, self._kernel_xm(x, np.array(self.x_obs[:-1])).reshape(self.n-1, 1)])
                    self.K = np.vstack([self.K, self._kernel_xm(x, np.array(self.x_obs)).reshape(1, self.n)])
                    # y_obs の更新
                    self.y_obs = np.append(self.y_obs, y)
                else:   #一度観測したことがある点という意味
                    index = self.map_obs[tuple(x)]
                    self.y_obs[index] = y
        else:
            # 初めてのself.dataなので以下をコンストラクト
            self.data = []
            for newdata in newdatas:
                self.data.append([[my_round0(newdata[0][0]), my_round0(newdata[0][1])], newdata[1]])
            self.x_obs = self._get_x_obs()
            self.n = len(self.x_obs)
            self.map_obs = self._get_map_obs()
            self.y_obs = self._get_y_obs()
            self.x_prd = self._get_x_prd()
            self.map_prd = self._get_map_prd()
            self.m = len(self.x_prd)
            self.K = self._get_K()
            self.k_star = self._get_k_star()
            self.k_2star = self._get_k_2star()