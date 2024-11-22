import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import os

DATA_FILE = "user_data.npy"

def get_user_input():
    if os.path.exists(DATA_FILE):
        states = list(np.load(DATA_FILE))
    else:
        states = []

    while len(states) < 100:
        state = input(f"请输入第{len(states) + 1}天的状态值（2: 一点欲望没有, 1: 有一点欲望, 0: 欲望适中, -1: 强烈的欲望）: ")
        try:
            state = int(state)
            if state in [2, 1, 0, -1]:
                states.append(state)
                np.save(DATA_FILE, np.array(states))
                print(f"状态 {state} 已成功提交")
            else:
                print("输入无效，请输入2, 1, 0, 或 -1。")
        except ValueError:
            print("输入无效，请输入整数值。")

    return np.array(states)

def train_model(data):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data.reshape(-1, 1))

    model = Sequential()
    model.add(Dense(10, input_dim=1, activation='relu'))  # 隐藏层
    model.add(Dense(1, activation='sigmoid'))  # 输出层

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model.fit(data, np.repeat([1], len(data)), epochs=500, verbose=1)

    return model, scaler

if __name__ == "__main__":
    user_data = get_user_input()

    if len(user_data) == 100:
        model, scaler = train_model(user_data)
        predicted_prob = model.predict(scaler.transform(user_data.reshape(-1, 1)))
        print(f"第100天的成功概率: {predicted_prob[-1][0] * 100:.2f}%")

        new_data = np.array([2, 1, -1, 0])
        new_data_scaled = scaler.transform(new_data.reshape(-1, 1))
        predicted_values = model.predict(new_data_scaled)
        print("每天状态值对成功概率的影响:")
        for i, prob in enumerate(predicted_values):
            print(f"状态值 {new_data[i]}: 预测成功概率为 {prob[0] * 100:.2f}%")
