import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import streamlit as st

# 构建神经网络模型
model = Sequential([
    Dense(10, input_dim=1, activation='relu'),
    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid')  # 输出值为 [0, 1]，表示成功概率
])

model.compile(optimizer='adam', loss='mse')

# 初始化数据
num_days = 100
progress = []  # 记录每天的状态
probability = []  # 记录每天的成功概率

# Streamlit 界面
st.title("戒色成功概率预测")
st.write("请每天更新你的戒色状态：")

# 获取用户输入
for day in range(num_days):
    status = st.slider(f"第 {day+1} 天的状态 (从 -2 到 2):", -2, 2, 0)
    progress.append([day])
    probability.append(status)

    # 将状态作为输入，训练模型并预测成功概率
    X = np.array(progress)
    y = np.array(probability).cumsum() / num_days  # 初始目标，实际会通过训练调整

    model.fit(X, y, epochs=10, verbose=0)
    pred_prob = model.predict(np.array([[day]]))[0][0]
    st.write(f"预测戒色成功概率: {pred_prob * 100:.2f}%")

# 可视化最终概率变化
fig, ax = plt.subplots()
ax.plot(range(1, num_days + 1), [model.predict(np.array([[i]]))[0][0] * 100 for i in range(num_days)], label='成功概率 (%)')
ax.set_xlabel('天数')
ax.set_ylabel('成功概率 (%)')
ax.set_title('戒色成功概率变化')
ax.legend()

st.pyplot(fig)
