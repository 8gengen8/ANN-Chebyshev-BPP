#模型：神经网络+切比雪夫基函数：监督标签
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from scipy import io
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error as mape
import matplotlib.pyplot as plt
plt.rcParams.update({'font.sans-serif': ['SimHei'], 'font.size': 12,'axes.unicode_minus': False})
#%% 1.混合模型
class BatchModel(nn.Module):
    #1)模型初始化
    def __init__(self, node_num, t, lb=0, ub=400):
        super().__init__()
        self.time = t
        #a.基函数权重映射网络
        layers = []
        for i in range(len(node_num)-1):
            if i == len(node_num)-2:
                layers.append(nn.Linear(node_num[i], node_num[i+1]))
            else:
                layers.extend([nn.Linear(node_num[i], node_num[i+1]), 
                        nn.LayerNorm(node_num[i+1]), nn.SiLU(), nn.Dropout(0.1)])
        self.model = nn.Sequential(*layers)
        self.basis_func = lambda t: torch.stack([
              torch.cos(i * torch.acos((2 * t.float() - (lb + ub)) / (ub - lb)))
              for i in range(node_num[-1])])
    #2)模型前向传播
    def forward(self, in_put):
        weight = self.model(in_put)
        basis_func_value = self.basis_func(self.time)
        out = weight @ basis_func_value
        return out
    #3)Adam优化器训练模型
    def train_adam(self, x_train, y_train, num_epochs=1000, lr=0.005):
        self.train()#必须设置训练模式
        optimizer = optim.Adam(self.parameters(),lr=lr)#配置优化器：自适应优化，使用默认超参
        #定义学习率：初始学习率lr*scheduler(scheduler方法限制)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs+1, eta_min=1e-10)
        best_loss = []
        criterion = nn.MSELoss()
        for epoch in range(num_epochs+1):
            epoch_loss = 0
            for batch_x, batch_y in dataloader:
                out = self(batch_x)
                loss = criterion(out, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()
            epoch_loss /= len(dataset)
            best_loss.append(epoch_loss)
            if epoch%100==0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'epoch:{epoch}/{num_epochs},Loss:{loss.item():.6e},LR: {current_lr:.6e}')
        return best_loss
    #4)模型预测
    def predict(self, x):
        self.eval()#必须设置评估模式
        with torch.no_grad():#关闭梯度，简化计算
            out_put=self(x)#调用forward前向计算
        return out_put
#%% 2.加载数据集
dataset = io.loadmat('../dataset.mat')
x_data, y_data, time = dataset['x_data'], dataset['y_data'], dataset['time'].flatten()
#%% 3.训练数据规范化
# 使用train_test_split划分数据集:7:3
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, 
                    test_size=0.3, random_state=42, shuffle=True)
# 将array数据转换成tensor数据
X_train = torch.tensor(x_train, dtype=torch.float32)
X_test = torch.tensor(x_test, dtype=torch.float32)
Y_train = torch.tensor(y_train, dtype=torch.float32)
Y_test = torch.tensor(y_test, dtype=torch.float32)
dataset = TensorDataset(X_train, Y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
print('数据切分完毕！')
for x_batch, y_batch in dataloader:
    print(f"特征批次形状: {x_batch.shape}, 目标批次形状: {y_batch.shape}")
    break
#%% 4.训练模型
node_num = [X_train.shape[1], 5, 12]#定义网络规模[..,24,16]
t = torch.tensor(time)
batch_model = BatchModel(node_num, t)#实例化模型
loss = batch_model.train_adam(X_train, Y_train,5001)#训练模型
#%%5.预测结果及性能指标
y_train_pre = batch_model.predict(X_train).detach().numpy()
y_test_pre = batch_model.predict(X_test).detach().numpy()
def counts(y_rea, y_pre, epsilon=0.01):
    error = y_rea - y_pre
    count1 = np.sum(np.abs(np.diff((error >= -epsilon) & (error <= epsilon))))
    region = np.zeros_like(error)
    region[error < -epsilon] = -1
    region[error > epsilon] = 1
    count2 = np.sum(np.abs(np.diff(region)) == 2)
    return (count1 + count2)/len(error)
MSE, MAE = mse(y_train_pre, y_train), mae(y_train_pre, y_train)
R2 = r2_score(y_train_pre.flatten(), y_train.flatten())
MAPE, ST = 100 * mape(y_train_pre, y_train), counts(y_train_pre, y_train)
met_train = [MSE, MAE, R2, MAPE, ST]
print(f'模型性能指标\n训练集：MSE={MSE:.4f}, MAE={MAE:.4f}, R2={R2:.4f}, MAPE={MAPE:.2f}%, ST={ST:.4f}')
MSE, MAE = mse(y_test_pre, y_test), mae(y_test_pre, y_test)
R2 = r2_score(y_test_pre.flatten(), y_test.flatten())
MAPE, ST = 100 * mape(y_test_pre, y_test), counts(y_test_pre, y_test)
print(f'测试集：MSE={MSE:.4f}, MAE={MAE:.4f}, R2={R2:.4f}, MAPE={MAPE:.2f}%, ST={ST:.4f}')
#%% 6.可视化展示
plt.figure()
for i in range(9,18):
    plt.subplot(3,3,i-8)
    plt.plot(time, y_train[i],'k-', time, y_train_pre[i],'r--', linewidth = 2.5)
    plt.yticks([0,0.5,1,1.5,2])
    plt.axis([0, 400, 0, 2])
    plt.xlabel('time:h')
    plt.ylabel('Pc：g/h')
    plt.legend(['Real','Pred'])
    plt.title(f'训练集预测结果：{i}批次')
plt.tight_layout()
plt.show()

plt.figure()
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.plot(time, y_test[i],'k-', time, y_test_pre[i],'r--', linewidth = 2.5)
    plt.yticks([0,0.5,1,1.5,2])
    plt.axis([0, 400, 0, 2])
    plt.xlabel('time:h')
    plt.ylabel('Pc：g/h')
    plt.legend(['Real','Pred'])
    plt.title(f'测试集预测结果：{i}批次')
plt.tight_layout()
plt.show()
#%% 7.保存结果数据
from scipy import io
result_our = {'y_train':y_train, 'y_train_pre':y_train_pre, 'y_test':y_test,
              'y_test_pre':y_test_pre, 'time':time, 'loss':loss, 'node_num':node_num}
io.savemat('result_our.mat', result_our)
#%%
plt.figure(num=3)
plt.title(f'Adam训练模型，best={min(loss):.4f}.')
plt.plot(loss)
plt.xlabel('迭代次数')
plt.ylabel('损失值')
plt.show()