import numpy as np

class KalmanFilter(object):
    def __init__(self, system_matrix, observation_matrix,
                 system_cov, observation_cov,
                 initial_state_mean, initial_state_cov,):
        self.system_matrix = np.array(system_matrix)
        self.observation_matrix = np.array(observation_matrix)
        self.system_cov = np.array(system_cov)
        self.observation_cov = np.array(observation_cov)
        self.initial_state_mean = np.array(initial_state_mean)
        self.initial_state_cov = np.array(initial_state_cov)
        
    def filter_predict(self, current_state_mean, current_state_cov):
        """フィルタリング分布から一期先予測分布(の平均と分散)を求める

        Args:
            current_state_mean (ndarray): フィルタリング分布の平均  
            current_state_cov (ndarray): フィルタリング分布の分散
        """
        predicted_state_mean = self.system_matrix @ current_state_mean
        predicted_state_cov = (
            self.system_matrix
            @ current_state_cov
            @ self.system_matrix.T
            + self.system_cov
        )
        return (predicted_state_mean, predicted_state_cov)

    def filter_update(self, predicted_state_mean, predicted_state_cov, observation):
        """一期先予測分布からフィルタリング分布(の平均と分散)を求める

        Args:
            predicted_state_mean (ndarray): 予測分布の平均
            predicted_state_cov (ndarray): 予測分布の分散
            observation (ndarray): 
        """
        kalman_gain = (
            predicted_state_cov 
            @ self.observation_matrix.T 
            @ np.linalg.inv(
                self.observation_matrix
                @ predicted_state_cov
                @ self.observation_matrix.T
                + self.observation_cov
            )
        )
        filtered_state_mean = (
            predicted_state_mean
            + kalman_gain
            @ (observation
            - self.observation_matrix
            @ predicted_state_mean)
        )
        filtered_state_cov = (
            predicted_state_cov
            - (kalman_gain
            @ self.observation_matrix
            @ predicted_state_cov)
        )
        return (filtered_state_mean, filtered_state_cov)

    def smooth_update(self, filtered_state_mean, filtered_state_cov,
                        predicted_state_mean, predicted_state_cov,
                        next_smoothed_state_mean, next_smoothed_state_cov):
        """時刻t+1の平滑化分布から1つ前の時刻tの平滑化分布を求める

        Args:
            filtered_state_mean (ndarray): t|tのフィルタ分布の平均
            filtered_state_cov (ndarray): t|tのフィルタ分布の分散
            predicted_state_mean (ndarray): t+1|tの予測分布の平均
            predicted_state_cov (ndarray): t+1|tの予測分布の分散
            next_smoothed_state_mean (ndarray): t|Tの平滑化分布の平均
            next_smoothed_state_cov (ndarray): t|Tの平滑化分布の分散

        Returns:
            tuple<ndarray>: t|Tの平滑化分布の平均と分散
        """
        kalman_smoothing_gain = (
            filtered_state_cov
            @ self.system_matrix.T
            @ np.linalg.inv(predicted_state_cov)
        )
        smoothed_state_mean = (
            filtered_state_mean
            + kalman_smoothing_gain
            @ (next_smoothed_state_mean - predicted_state_mean)
        )
        smoothed_state_cov = (
            filtered_state_cov
            + kalman_smoothing_gain
            @ (next_smoothed_state_cov - predicted_state_cov)
            @ kalman_smoothing_gain.T
        )
        return (smoothed_state_mean, smoothed_state_cov)

    def filter(self, observations):
        """フィルタリング

        Args:
            observations (list): 観測系列データ

        Returns:
            tuple<ndarray bi 4>: 各時刻全てのフィルタリング分布(の平均と分散)と予測分布(の平均と分散)のコンテナ
        """
        observations = np.array(observations)
        
        n_timesteps = len(observations)
        n_dim_state = len(self.initial_state_mean)
        
        predicted_state_means = np.zeros((n_timesteps, n_dim_state))
        predicted_state_covs = np.zeros((n_timesteps, n_dim_state, n_dim_state))
        filtered_state_means = np.zeros((n_timesteps, n_dim_state))
        filtered_state_covs = np.zeros((n_timesteps, n_dim_state, n_dim_state))
        
        for t in range(n_timesteps):
            if t == 0:
                predicted_state_means[t] = self.initial_state_mean
                predicted_state_covs[t] = self.initial_state_cov
            else:
                predicted_state_means[t], predicted_state_covs[t] = self.filter_predict(
                    filtered_state_means[t-1],
                    filtered_state_covs[t-1]
                )
            filtered_state_means[t], filtered_state_covs[t] = self.filter_update(
                predicted_state_means[t],
                predicted_state_covs[t],
                observations[t]
            )
        
        return (
            filtered_state_means,
            filtered_state_covs,
            predicted_state_means,
            predicted_state_covs
        )
        
    def predict(self, n_timesteps, observations):
        """(n期先)予測

        Args:
            n_timesteps (int): どのくらい先の予測を行うか
            observations (list): 観測系列データ

        Returns:
            tuple<ndarray>: n期先までのフィルタリング分布(の平均と分散)と予測分布(の平均と分散)のコンテナ
        """
        (filtered_state_means,
        filtered_state_covs,
        predicted_state_means,
        predicted_state_covs) = self.filter(observations)

        _, n_dim_state = filtered_state_means.shape

        predicted_state_means = np.zeros((n_timesteps, n_dim_state))
        predicted_state_covs = np.zeros((n_timesteps, n_dim_state, n_dim_state))

        for t in range(n_timesteps):
            if t == 0:
                predicted_state_means[t], predicted_state_covs[t] = self.filter_predict(
                    filtered_state_means[-1],
                    filtered_state_covs[-1]
                )
            else:
                predicted_state_means[t], predicted_state_covs[t] = self.filter_predict(
                    predicted_state_means[t-1],
                    predicted_state_covs[t-1]
                )

        return (predicted_state_means, predicted_state_covs)
    
    def smooth(self, observations):
        """平滑化

        Args:
            observations (list): 観測系列データ

        Returns:
            tuple<ndarray>: 時刻Tまで全ての平滑化分布の平均と分散
        """
        (filtered_state_means,
        filtered_state_covs,
        predicted_state_means,
        predicted_state_covs) = self.filter(observations)

        n_timesteps, n_dim_state = filtered_state_means.shape

        smoothed_state_means = np.zeros((n_timesteps, n_dim_state))
        smoothed_state_covs = np.zeros((n_timesteps, n_dim_state, n_dim_state))

        smoothed_state_means[-1] = filtered_state_means[-1]
        smoothed_state_covs[-1] = filtered_state_covs[-1]

        for t in reversed(range(n_timesteps - 1)):
            smoothed_state_means[t], smoothed_state_covs[t] = self.smooth_update(
                filtered_state_means[t],
                filtered_state_covs[t],
                predicted_state_means[t+1],
                predicted_state_covs[t+1],
                smoothed_state_means[t+1],
                smoothed_state_covs[t+1]
            )
        
        return (smoothed_state_means, smoothed_state_covs)
    
import random as rd

def generateT(length):
    dt = 1
    pos, a_right = [], []
    px, py, vx, vy, ax, ay = 0, 0, 0, 0, 0, 0
    pos.append([px, py])
    a_right.append([ax, ay])
    for t in range(length):
        if t < length*0.3:
            ax += .2
            ay += .2
        else:
            ax += .1 
            ay -= .1 
        vx = vx +  ax * dt
        vy = vy +  ay * dt
        px += vx * dt + ax * dt**2 / 2
        py += vy * dt + ay * dt**2 / 2
        zx = rd.uniform(-vx, vx)
        zy = rd.uniform(-vy, vy)
        pos.append([px, py])
        a_right.append([ax, ay])
    return pos, a_right

"""
system_matrix = [[1, 1, 1/2],[0, 1, 1],[0, 0, 1]]
observation_matrix = [[1, 0, 0]]
system_cov = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
observation_cov = [1]
initial_state_mean = [0, 0, 0]
initial_state_cov = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
track, _ = generateT(10)
kalman = KalmanFilter(system_matrix, observation_matrix, system_cov, observation_cov, initial_state_mean, initial_state_cov)
print(kalman.predict(1, track))
"""
"""
dt = 1
system_matrix = [[1, 0, dt, 0, dt**2/2, 0],
                    [0, 1, 0, dt, 0, dt**2/2],
                    [0, 0, 1, 0, dt, 0],
                    [0, 0, 0, 1, 0, dt],
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
initial_state_cov = [[1, 0, 0, 0, 0, 0], 
                        [0, 1, 0, 0, 0, 0], 
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1]
                        ]
track, _  = generateT(1)
initial_state_mean = [track[0][0], track[0][1], (track[1][0]-track[0][0])/dt, (track[1][1]-track[0][1])/dt, 0, 0]
kalman = KalmanFilter(system_matrix, observation_matrix, system_cov, observation_cov, initial_state_mean, initial_state_cov)
state_mean, state_cov = kalman.predict(1, track[1:])
print(track[1:])
state_mean = state_mean.reshape(6)
for p in track:
    print(p)
print(state_mean[:2])
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([row[0] for row in track], [row[1] for row in track], color="red")
ax.plot([state_mean[0], track[-1][0]], [state_mean[1], track[-1][1]], color="blue")
fig.savefig("misc.pdf")
"""
