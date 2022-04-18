import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

OUT_DIR = './CNN_out'  # 저장 폴더 지정

img_shape = (28, 28, 1)  # 이미지 모양
epochs = 100000
batch_size = 128
noise = 100
sample_interval = 100

(X_train, _), (_, _) = mnist.load_data()  #mnist 트레인 파일 가져옴
print(X_train.shape)

X_train = X_train / 127.5 - 1 # 범위를 -1 ~ 1사이로 지정 , 옅은 색은 학습이 덜 되게 해라
X_train = np.expand_dims(X_train, axis=3)
print(X_train.shape)


generator = Sequential()
generator.add(Dense(256*7*7, input_dim=noise))
generator.add(Reshape((7, 7, 256))) #  7,7 사진을 256장 리쉡
generator.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same')) # 256장을 128장으로 줄여나감, 커널 3,3으로 128로 줄임
generator.add(BatchNormalization()) #  256을 128장으로 줄여서 값이 2배로 커지는 값을 원래 1장의 값의 평균 값으로 변경
generator.add(LeakyReLU(alpha=0.01))

generator.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))  # 위에 과정 한번 더
generator.add(BatchNormalization())
generator.add(LeakyReLU(alpha=0.01))

generator.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same'))
generator.add(Activation('tanh')) #generator 끝

discriminator = Sequential()
discriminator.add(Conv2D(32, kernel_size=3,
        strides=2, padding='same', input_shape=img_shape))
discriminator.add(LeakyReLU(alpha=0.01))
discriminator.add(Conv2D(64, kernel_size=3,
        strides=2, padding='same'))
discriminator.add(LeakyReLU(alpha=0.01))
discriminator.add(Conv2D(128, kernel_size=3,
        strides=2, padding='same'))
discriminator.add(LeakyReLU(alpha=0.01))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.summary()

discriminator.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
discriminator.trainable=False

gan_model = Sequential()
gan_model.add(generator)
gan_model.add(discriminator)
gan_model.summary()
gan_model.compile(loss='binary_crossentropy', optimizer='adam')

real = np.ones((batch_size, 1))

fake = np.zeros((batch_size, 1))

for epoch in range(epochs):
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_imgs = X_train[idx]

    z = np.random.normal(0, 1, (batch_size, noise))
    fake_imgs = generator.predict(z)

    d_hist_real = discriminator.train_on_batch(real_imgs, real)
    d_hist_fake = discriminator.train_on_batch(fake_imgs, fake)

    d_loss, d_acc = 0.5 * np.add(d_hist_real, d_hist_fake)
    discriminator.trainable=False

    z = np.random.normal(0, 1, (batch_size, noise))
    gan_hist = gan_model.train_on_batch(z, real)

    if epoch % sample_interval == 0:
        print('%d [D loss: %f, acc.: %.2f%%] [G loss: %f]'%(
            epoch, d_loss, d_acc * 100, gan_hist))
        row = col = 4
        z = np.random.normal(0, 1, (row * col, noise))
        fake_imgs = generator.predict(z)
        fake_imgs = 0.5 * fake_imgs + 0.5
        _, axs = plt.subplots(row, col, figsize=(row, col),
                             sharey=True, sharex=True)
        cont = 0
        for i in range(row):
            for j in range(col):
                axs[i, j].imshow(fake_imgs[cont, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cont += 1

        generator.save('./content/drive/MyDirve/generator_mnist_{}.h5'.format(MY_NUMBER))


