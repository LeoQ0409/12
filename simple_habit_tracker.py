import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import os
import matplotlib.pyplot as plt

# 准备数据
DATA_FILE = "user_data.npy"

# Streamlit App
st.title("戒色状态跟踪应用")

if os.path.exists(DATA_FILE):
    states = list(np.load(DATA_FILE))
else:
    states = []

# 获取用户输入
state = st.number_input("请输入今天的状态值 (2: 一点欲望没有, 1: 有一点欲望, 0: 欲望适中, -1: 强烈的欲望):", min_value=-1, max_value=2, step=1)

if st.button("提交状态"):
    states.append(state)
    np.save(DATA_FILE, np.array(states))
    st.success(f"状态 {state} 已成功提交")

if len(states) > 0:
    st.write("当前已记录的状态值:", states)

# 创建神经网络模型并进行训练
if len(states) >= 10:
    scaler = MinMaxScaler()
    data = scaler.fit_transform(np.array(states).reshape(-1, 1))

    model = Sequential()
    model.add(Dense(10, input_dim=1, activation='relu'))  # 隐藏层，有10个神经元
    model.add(Dense(1, activation='sigmoid'))  # 输出层，输出值在0到1之间

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    # 训练模型
    model.fit(data, np.repeat([1], len(data)), epochs=500, verbose=0)

    # 测试模型
    predicted_prob = model.predict(data)
    st.write(f"第{len(states)}天的成功概率: {predicted_prob[-1][0] * 100:.2f}%")

    # 绘制概率图
    fig, ax = plt.subplots()
    ax.plot(range(1, len(states) + 1), predicted_prob * 100, marker='o')
    ax.set_xlabel('天数')
    ax.set_ylabel('成功概率 (%)')
    ax.set_title('成功概率随天数的变化')
    ax.grid(True)
    st.pyplot(fig)

    # 使用模型来预测每天的状态值对成功概率的影响
    new_data = np.array([2, 1, -1, 0])  # 示例输入
    new_data = scaler.transform(new_data.reshape(-1, 1))
    predicted_values = model.predict(new_data)
    st.write("每天状态值对成功概率的影响:")
    for i, prob in enumerate(predicted_values):
        st.write(f"状态值 {new_data[i][0]}: 预测成功概率为 {prob[0] * 100:.2f}%")
