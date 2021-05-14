import numpy as np

(X_train, _) , (_,_) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5

X_train = X_train.reshape(60000,784)

def train(epochs=1, batchSize=128):
    batchCount = int(X_train.shape[0] / batchSize)
    print('Epochs : ', epochs)
    print('Batch Size : ', batchSize)
    print('Batch per epoch : ', batchCount)

    for e in range(1,epochs+1):
        print('-'*15, 'Epoch %d' %e, '-'*15)
        for _ in range(batchCount):
            noise = np.random.normal(0,1,size=[batchSize,randomDim])
            imageBatch = X_train[np.random.randint(0,X_train.shape[0],size=batchSize)]

            generatedImages = generator.predict(noise)


        X = np.concatenate([imageBatch,generatedImages])

        #생성된 것과 실제 이미지의 레이블
        yDis = np.zeros(2*batchSize)
        #편파적 레이블 평활화
        yDis[:batchSize] = 0.9

        #discriminator 훈련
        discriminator.trainable = True
        dloss = discriminator.tarin_on_batch(X,yDis)

        #generator 훈련
        noise = np.random.normal(0,1,size=[batchSize, randomDim])
        yGen = np.ones(batchSize)
        discriminator.trainable = False
        gloss = gan.train_on_batch(noise, yGen)

    dLosses.append(dloss)
    gLosses.append(gloss)

    if e == 1 or e % 20 == 0:
        saveGeneratedImages(e)
    

