import numpy
numpy.random.seed(0)

import CDAE_model_establishment
import input_dataset_movie_lens


# data
train_users, train_x, test_users, test_x = input_dataset_movie_lens.load_data()
train_x_users = numpy.array(train_users, dtype=numpy.int32).reshape(len(train_users), 1)
test_x_users = numpy.array(test_users, dtype=numpy.int32).reshape(len(test_users), 1)

# model
model = CDAE_model_establishment.create(I=train_x.shape[1], U=len(train_users)+1, K=50,
                    hidden_activation='relu', output_activation='sigmoid', q=0.50, l=0.01)
model.compile(loss='mean_absolute_error', optimizer='adam')
model.summary()

# train
history = model.fit(x=[train_x, train_x_users], y=train_x,
                    batch_size=128, nb_epoch=1000, verbose=1,
                    validation_data=[[test_x, test_x_users], test_x])

# predict
pred = model.predict(x=[train_x, numpy.array(train_users, dtype=numpy.int32).reshape(len(train_users), 1)])
pred = pred * (train_x == 0) # remove watched items from predictions
pred = numpy.argsort(pred)



import numpy

def success_rate(pred, true):
    cnt = 0
    for i in range(pred.shape[0]):
        t = numpy.where(true[i] == 1) # true set
        ary = numpy.intersect1d(pred[i], t)
        if ary.size > 0:
            cnt += 1
    return cnt * 100 / pred.shape[0]


for n in range(1, 11):
    sr = success_rate(pred[:, -n:], test_x)
    print("Success Rate at {:d}: {:f}".format(n, sr))


