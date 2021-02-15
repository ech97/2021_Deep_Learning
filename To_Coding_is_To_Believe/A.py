from tensorflow.keras.datasets.mnist import load_data

(x_train, y_train), (x_test, y_test) = load_data(path='mnist.npz')

# 입력층 확인

print(x_train.shape, y_train.shape)
print(y_train)

print(x_test.shape, y_test.shape)
print(y_test)


# 그래프 그려보기
import matplotlib.pyplot as plt
import numpy as np

sample_size = 3
random_idx = np.random.randint(60000, size=sample_size)

for idx in random_idx:
    img = x_train[idx, :]
    label = y_train[idx]
    plt.figure()
    plt.imshow(img)
    plt.title(f"{idx} in data, label is {label}")



# 검증 데이터 만들어주기 / train에서 일부를 떼어 val로 만들어주는작업

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val  = train_test_split(x_train, y_train, test_size = 0.3, random_state = 777)  # 시드 고정

print(f"훈련 데이터 {x_train.shape} 레이블 {y_train.shape}")
print(f"검증 데이터 {x_val.shape} 레이블 {y_val.shape}")



# 전처리 과정 (모델에 맞게 이미지 조정)

num_x_train = x_train.shape[0] # 42000개
num_x_val = x_val.shape[0] # 18000개
num_x_test = x_test.shape[0] # 10000개

x_train = (x_train.reshape((num_x_train, 28*28))) / 255 # 원래는 (42000, 28, 28)이었으나 (42000, 784)로 모델에 맞게 변경
x_val = (x_val.reshape((num_x_val, 28*28))) / 255
x_test = (x_test.reshape((num_x_test, 28*28))) / 255

print(x_train.shape)