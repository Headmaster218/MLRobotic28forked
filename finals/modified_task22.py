import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 加载数据
with open('finals/data.pkl', 'rb') as file:
    data = pickle.load(file)

# 数据准备
time = np.array(data['time']).reshape(-1, 1)  # 时间步长
goal_positions = np.array(data['goal_positions'])  # 目标位置 (x, y, z)
q_mes_all = np.array(data['q_mes_all'])  # 关节位置 (7 个关节)

# 输入特征和输出目标
input_data = np.hstack((time, goal_positions))  # 合并时间和目标位置为输入
output_data = q_mes_all  # 输出为关节位置

# 训练和测试集划分
split_index = int(0.8 * len(input_data))
train_X, test_X = input_data[:split_index], input_data[split_index:]
train_y, test_y = output_data[:split_index], output_data[split_index:]

# 定义函数以不同深度训练随机森林模型并绘制结果
def train_and_evaluate_rf(max_depth):
    # 定义并训练随机森林模型
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=max_depth, random_state=0)
    rf_model.fit(train_X, train_y)

    # 计算训练和测试损失
    train_predictions = rf_model.predict(train_X)
    test_predictions = rf_model.predict(test_X)
    train_loss = mean_squared_error(train_y, train_predictions)
    test_loss = mean_squared_error(test_y, test_predictions)

    print(f"Random Forest (max_depth={max_depth}) - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    # 绘制测试集关节位置预测轨迹
    plot_joint_trajectories(test_y, test_predictions, title=f"Random Forest Predictions (max_depth={max_depth})")

# 定义绘图函数
def plot_joint_trajectories(actual, predicted, title="Joint Trajectories"):
    num_joints = actual.shape[1]
    plt.figure(figsize=(12, 8))
    for i in range(num_joints):
        plt.subplot(3, 3, i + 1)
        plt.plot(actual[:, i], label="Actual", linestyle='--')
        plt.plot(predicted[:, i], label="Predicted")
        plt.title(f"Joint {i+1}")
        plt.xlabel("Time Step")
        plt.ylabel("Position")
        plt.legend()
    
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# 测试不同的树深度
for depth in [None, 2, 5, 10]:
    train_and_evaluate_rf(max_depth=depth)
