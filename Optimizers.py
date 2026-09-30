#优化器列表
#1.标准金枪鱼优化器TSO(实数域优化)：Improved_Tuna_Swarm_Optimization
#2.标准金枪鱼优化器ITSO(正整数域优化)ITSO
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
#%%1.标准金枪鱼优化器TSO(实数域优化)
class Tuna_Swarm_Optimization:
    """
    Standard Tuna Swarm Optimization (ITSO) 算法实现
    参考论文：Tuna Swarm Optimization: A Novel Swarm-Based Metaheuristic Algorithm for Global Optimization
    https://onlinelibrary.wiley.com/doi/10.1155/2021/9210050
    #%% 示例使用:正整数域最小化Sphere函数, x*=[0,0,...,0]
    if __name__ == "__main__":
        from Optimizers import Tuna_Swarm_Optimization
        def sphere(x):
            return np.sum(x**2 + np.abs(10*np.sin(x)))+5
        # 定义搜索空间
        lb = -20*np.array([1,1,1])
        ub = 20*np.array([1,1,1])
        dim = len(lb)
        TSO = Tuna_Swarm_Optimization()
        best_pos, best_fit = ITSO.optimize(sphere, lb, ub, swarmsize=50, maxiter=500)
        # 绘制收敛曲线
        itso_loss = ITSO.plot_convergence()
    """
    #优化主函数
    def optimize(self, func, lb, ub, swarmsize=80, maxiter= 200, a= 0.7, z= 0.05):
        """
        ITSO算法优化目标函数
        参数:
        func: 可调用的目标函数
        ub, lb: 搜索空间上下界
        swarmsize: 种群规模
        maxiter: 最大迭代次数
        a, z: TSO算法参数
        返回:
        best_part: 最优解位置
        best_fit: 最优适应度值
        """
        # 初始化参数
        self.func = func
        self.lb, self.ub = lb, ub
        self.dim = len(lb)
        self.swarmsize = swarmsize
        self.maxiter = maxiter
        self.a, self.z = a, z
        # 初始化种群
        self.part = self._initialize_population()
        self.fit = [self.func(part) for part in self.part]
        # 初始化历史记录
        self.history = {'best_part': None, 'best_fit': None, 'fit': [], }
        # 寻找初始最优解
        best_idx = np.argmin(self.fit)
        self.history['best_part'] = self.part[best_idx]
        self.history['best_fit'] = self.fit[best_idx]
        self.history['fit'].append(self.history['best_fit'])
        
        # 优化过程
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}'
        with tqdm(total=maxiter, desc = 'ITSO train model', bar_format = bar_format) as pbar:
            for iter_num in range(maxiter):
                #更新个体位置和适应度值
                self._update_population(iter_num + 1)
                best_idx = np.argmin(self.fit)
                if self.fit[best_idx] < self.history['best_fit']:
                    # 更新历史记录：必须使用copy属性，防止在else被更新
                    self.history['best_part'] = self.part[best_idx].copy()
                    self.history['best_fit'] = self.fit[best_idx].copy()
                    self.history['fit'].append(self.fit[best_idx])
                else:
                    self.history['fit'].append(self.history['fit'][-1])
                # 动态更新进度条后缀信息
                pbar.set_postfix(best_loss=f"{self.history['best_fit']}",
                                 best_part=f"{self.history['best_part']}")
                pbar.update(1)
        return self.history['best_part'], self.history['best_fit']
    #初始化种群
    def _initialize_population(self):
        """初始化种群"""
        return np.random.uniform(self.lb, self.ub, (self.swarmsize, self.dim))
    #位置更新函数
    def _update_population(self, iter_num):
        """更新种群位置"""
        for i in range(self.swarmsize):
            if np.random.rand() < self.z:
                new_position = np.random.uniform(self.lb, self.ub, self.dim)
            else:
                if np.random.rand() < 0.5:
                    # 螺旋觅食策略
                    alpha1 = self.a + (1 - self.a) * iter_num / self.maxiter
                    alpha2 = (1 - self.a) * (1 - iter_num / self.maxiter)
                    b = np.random.rand()
                    l = np.exp(3 * np.cos((self.maxiter + iter_num) / iter_num * np.pi))
                    beta = np.exp(b * l)*np.cos(2* np.pi * b)
                    if np.random.rand() > iter_num / self.maxiter:
                        position_shift = np.random.uniform(self.lb, self.ub, self.dim)
                    else:
                        position_shift = self.history['best_part']
                    if i == 0:
                        new_position = alpha1 * (position_shift + beta * np.abs(
                            position_shift - self.part[i])) + alpha2 * self.part[i]
                    else:
                        new_position = alpha1 * (position_shift + beta * np.abs(
                            position_shift - self.part[i])) + alpha2 * self.part[i-1]
                else:
                    # 抛物线觅食策略
                    p=(1 - iter_num / self.maxiter)**(iter_num / self.maxiter)
                    TF = 1 if np.random.rand() > 0.5 else -1
                    if np.random.rand() < 0.5:
                        # 第一种抛物线觅食
                        new_position = self.history['best_part'] + np.random.randn(self.dim) * (
                            self.history['best_part'] - self.part[i]) + TF * p**2 * (
                            self.history['best_part'] - self.part[i])
                    else:
                        # 第二种抛物线觅食
                        new_position = TF * p**2 * self.part[i]
            
            # 约束+边界处理形成新位置, 并计算其适应度值
            self.part[i] = np.clip(new_position, self.lb, self.ub)
            self.fit[i] = self.func(self.part[i])
    #可视化函数
    def plot_convergence(self):
        """绘制收敛曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['fit'][1:], 'b-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness')
        plt.title('ITSO Convergence Curve')
        plt.grid(True)
        plt.show()
        return self.history['fit']

#%%2.标准金枪鱼优化器ITSO(正整数域优化)
class Improved_Tuna_Swarm_Optimization:
    """
    Improved Tuna Swarm Optimization (ITSO) 算法实现：正整数域优化
    #%% 示例使用:正整数域最小化Sphere函数, x*=[1,1,...,1]
    if __name__ == "__main__":
        from Optimizers import Improved_Tuna_Swarm_Optimization
        def sphere(x):
            return np.sum(x**2 + np.abs(10*np.sin(x)))+5
        # 定义搜索空间
        lb = -20*np.array([1,1,1])
        ub = 20*np.array([1,1,1])
        dim = len(lb)
        ITSO = Improved_Tuna_Swarm_Optimization()
        best_pos, best_fit = ITSO.optimize(sphere, lb, ub, swarmsize=50, maxiter=500)
        # 绘制收敛曲线
        itso_loss = ITSO.plot_convergence()
    """
    #优化主函数
    def optimize(self, func, lb, ub, swarmsize=80, maxiter= 200, a= 0.7, z= 0.05):
        """
        ITSO算法优化目标函数
        参数:
        func: 可调用的目标函数
        ub, lb: 搜索空间上下界
        swarmsize: 种群规模
        maxiter: 最大迭代次数
        a, z: TSO算法参数
        返回:
        best_part: 最优解位置
        best_fit: 最优适应度值
        """
        # 初始化参数
        self.func = func
        self.lb, self.ub = lb, ub
        self.dim = len(lb)
        self.swarmsize = swarmsize
        self.maxiter = maxiter
        self.a, self.z = a, z
        # 初始化种群
        self.part = self.trans_func(self._initialize_population())
        self.fit = [self.func(part) for part in self.part]
        # 初始化历史记录
        self.history = {'best_part': None, 'best_fit': None, 'fit': [], }
        # 寻找初始最优解
        best_idx = np.argmin(self.fit)
        self.history['best_part'] = self.part[best_idx]
        self.history['best_fit'] = self.fit[best_idx]
        self.history['fit'].append(self.history['best_fit'])
        
        # 优化过程
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}'
        with tqdm(total=maxiter, desc = 'ITSO train model', bar_format = bar_format) as pbar:
            for iter_num in range(maxiter):
                #更新个体位置和适应度值
                self._update_population(iter_num + 1)
                best_idx = np.argmin(self.fit)
                if self.fit[best_idx] < self.history['best_fit']:
                    # 更新历史记录：必须使用copy属性，防止在else被更新
                    self.history['best_part'] = self.part[best_idx].copy()
                    self.history['best_fit'] = self.fit[best_idx].copy()
                    self.history['fit'].append(self.fit[best_idx])
                else:
                    self.history['fit'].append(self.history['fit'][-1])
                # 动态更新进度条后缀信息
                pbar.set_postfix(best_loss=f"{self.history['best_fit']}",
                                 best_part=f"{self.history['best_part']}")
                pbar.update(1)
        return self.history['best_part'], self.history['best_fit']
    #初始化种群
    def _initialize_population(self):
        """初始化种群"""
        return np.random.uniform(self.lb, self.ub, (self.swarmsize, self.dim))
    def trans_func(self, part):
        return np.floor(np.abs(part) + 0.5) + 1
    #位置更新函数
    def _update_population(self, iter_num):
        """更新种群位置"""
        for i in range(self.swarmsize):
            if np.random.rand() < self.z:
                new_position = np.random.uniform(self.lb, self.ub, self.dim)
            else:
                if np.random.rand() < 0.5:
                    # 螺旋觅食策略
                    alpha1 = self.a + (1 - self.a) * iter_num / self.maxiter
                    alpha2 = (1 - self.a) * (1 - iter_num / self.maxiter)
                    b = np.random.rand()
                    l = np.exp(3 * np.cos((self.maxiter + iter_num) / iter_num * np.pi))
                    beta = np.exp(b * l)*np.cos(2* np.pi * b)
                    if np.random.rand() > iter_num / self.maxiter:
                        position_shift = np.random.uniform(self.lb, self.ub, self.dim)
                    else:
                        position_shift = self.history['best_part']
                    if i == 0:
                        new_position = alpha1 * (position_shift + beta * np.abs(
                            position_shift - self.part[i])) + alpha2 * self.part[i]
                    else:
                        new_position = alpha1 * (position_shift + beta * np.abs(
                            position_shift - self.part[i])) + alpha2 * self.part[i-1]
                else:
                    # 抛物线觅食策略
                    p=(1 - iter_num / self.maxiter)**(iter_num / self.maxiter)
                    TF = 1 if np.random.rand() > 0.5 else -1
                    if np.random.rand() < 0.5:
                        # 第一种抛物线觅食
                        new_position = self.history['best_part'] + np.random.randn(self.dim) * (
                            self.history['best_part'] - self.part[i]) + TF * p**2 * (
                            self.history['best_part'] - self.part[i])
                    else:
                        # 第二种抛物线觅食
                        new_position = TF * p**2 * self.part[i]
            
            # 约束+边界处理形成新位置, 并计算其适应度值
            self.part[i] = np.clip(self.trans_func(new_position), self.lb, self.ub)
            self.fit[i] = self.func(self.part[i])
    #可视化函数
    def plot_convergence(self):
        """绘制收敛曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['fit'][1:], 'b-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness')
        plt.title('ITSO Convergence Curve')
        plt.grid(True)
        plt.show()
        return self.history['fit']