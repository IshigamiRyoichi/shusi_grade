import cv2
import numpy as np
from keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
import glob

# ラベル一覧
label_list = ["6/0/","6/1/","17/0/","17/1/","18/0/","18/1/","19/0/","19/1/","27/0/","27/1/","41/0/","41/1/","81/0/","81/1/"]

# 教師データのラベル
X_train = []
y_train = []
# 判定用
train_label = 0

# ラベル事に処理
for label in label_list:
    image_file_list = glob.glob("./attach/sushi_m_r/" + label + "*.jpg")
    for image_file in image_file_list:
        image_data = cv2.imread(image_file)
        #色成分を分割
        b,g,r = cv2.split(image_data)
        #色成分を結合
        image_data = cv2.merge([r,g,b])
        X_train.append(image_data)
        y_train.append(train_label)
    train_label += 1

# テストデータのラベル
X_test = []
y_test = []
# 判定用
test_label = 0

# ラベル事に処理
for label in label_list:
    image_file_list = glob.glob("./attach/sushi_s_r/" + label + "*.jpg")
    for image_file in image_file_list:
        image_data = cv2.imread(image_file)
        #色成分を分割
        b,g,r = cv2.split(image_data)
        #色成分を結合
        image_data = cv2.merge([r,g,b])
        X_test.append(image_data)
        y_test.append(test_label)
    test_label += 1

#配列化
X_train=np.array(X_train)
X_test=np.array(X_test)

print()

#ラベルをone-hotベクトルにする？
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# モデルの定義
model = Sequential()
#畳み込みオートエンコーダーの動作
#ここの64は画像サイズ
#画像サイズがあっていないと、エラーが発生する
#3×3のフィルターに分ける
model.add(Conv2D(input_shape=(64, 64, 3), filters=32,kernel_size=(3, 3), 
                 strides=(1, 1), padding="same"))
#2×2の範囲で最大値を出力
model.add(MaxPooling2D(pool_size=(2, 2)))
#畳み込みオートエンコーダーの動作
#3×3のフィルターに分ける
model.add(Conv2D(filters=32, kernel_size=(3, 3), 
                 strides=(1, 1), padding="same"))
#2×2の範囲で最大値を出力
model.add(MaxPooling2D(pool_size=(2, 2)))
#畳み込みオートエンコーダーの動作
#3×3のフィルターに分ける
model.add(Conv2D(filters=32, kernel_size=(3, 3), 
                 strides=(1, 1), padding="same"))
#2×2の範囲で最大値を出力
model.add(MaxPooling2D(pool_size=(2, 2)))
#1次元配列に変換
model.add(Flatten())
#出力の次元数を256にする
model.add(Dense(256))
#非線形変形の処理
model.add(Activation("sigmoid"))
#出力の次元数を128にする
model.add(Dense(128))
#非線形変形の処理をするらしい
model.add(Activation('sigmoid'))
#出力の次元数を14にする
model.add(Dense(14))
#非線形変形の処理をするらしい
model.add(Activation('softmax'))

# コンパイル
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# 学習
history = model.fit(X_train, y_train, batch_size=32, 
                    epochs=45, verbose=1, validation_data=(X_test, y_test))

# 汎化制度の評価・表示
score = model.evaluate(X_test, y_test, batch_size=32, verbose=0)
print('validation loss:{0[0]}\nvalidation accuracy:{0[1]}'.format(score))

#モデルを保存
model.save("./sushi_gread.h5")