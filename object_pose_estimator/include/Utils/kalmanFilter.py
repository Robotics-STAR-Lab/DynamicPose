import numpy as np

class KalmanFilter:
    def __init__(self, shape=6):
        # self.A = A # 状态转移
        # self.B = B # 控制矩阵
        # self.H = H # 观测矩阵
        # self.Q = Q # 过程噪声协方差
        # self.R = R # 观测噪声协方差
        # self.P = P # 误差协方差
        # self.x = x0 # 初始状态
        self.defaultInit(shape)
    
    def defaultInit(self, shape):
        self.A = np.eye(shape)
        self.B = np.eye(shape)
        self.H = np.eye(shape)
        self.Q = np.eye(shape) * 0.5
        self.R = np.eye(shape) * 0.5
        self.P = np.eye(shape) * 0.1
        # self.x = np.zeros(6)
        
    def predict(self):
        u = np.zeros(self.x.shape)
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
    
    def update(self, z):
        # 计算卡尔曼增益
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # 更新估计值
        self.x = self.x + np.dot(K, z - np.dot(self.H, self.x))
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)
        
        return self.x
    
    def reset(self, x0):
        self.x = x0
        self.defaultInit(self.x.shape[0])