import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
from PIL import Image

def sigmoid(x):
    result = 1 / (1 + np.exp(-x))
    return result

def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0

    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b

def propagate(w, b, X, Y):
    m = X.shape[1] # 训练数据个数
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
    cost = - 1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    dw = 1 / m * np.dot(X, (A - Y).T)
    db = 1 / m * np.sum(A - Y)

    ### END CODE HERE ###
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
            # Print the cost every 100 training examples
            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs

def predict(w, b, X):
    m = X.shape[1]
    Y_hat = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        if A[0, i] <= 0.5:
            Y_hat[0, i] = 0
        else:
            Y_hat[0, i] = 1

    assert (Y_hat.shape == (1, m))
    return  Y_hat


def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    w, b = initialize_with_zeros(X_train.shape[0])

    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w = params["w"]
    b = params["b"]

    Y_train_hat =  predict(w, b, X_train)
    Y_test_hat = predict(w, b, X_test)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_train_hat - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_test_hat - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_test_hat,
         "Y_prediction_train": Y_train_hat,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


def train_process(x, y, w, b):
    epoch = 100
    m = x.shape[1]
    learning_rate = 0.009
    
    for i in range(epoch):
        z = np.dot(w.T, x) + b
        a = sigmoid(z)
        loss = - 1 / m * np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))
        print(str(i) + " epoch loss: " + str(loss))
        dw = 1 / m * np.dot(x, (a - y).T)
        db = 1 / m * np.sum(a - y)
        w = w - learning_rate * dw
        b = b - learning_rate * db
    return w, b


if __name__ == '__main__':
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]
    # print("Number of training examples: m_train = " + str(m_train))
    # print("Number of testing examples: m_test = " + str(m_test))
    # print("Height/Width of each image: num_px = " + str(num_px))
    # print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    # print("train_set_x shape: " + str(train_set_x_orig.shape))
    # print("train_set_y shape: " + str(train_set_y.shape))
    # print("test_set_x shape: " + str(test_set_x_orig.shape))
    # print("test_set_y shape: " + str(test_set_y.shape))

    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    # print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
    # print("train_set_y shape: " + str(train_set_y.shape))
    # print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
    # print("test_set_y shape: " + str(test_set_y.shape))
    # print("sanity check after reshaping: " + str(train_set_x_flatten[0:5, 0]))

    train_set_x = train_set_x_flatten / 255.
    test_set_x = test_set_x_flatten / 255.

    d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005,
              print_cost=True)

    # index = 1
    # plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
    # print("y = " + str(test_set_y[0, index]) + ", you predicted that it is a \"" + classes[
    #     int(d["Y_prediction_test"][0, index])].decode("utf-8") + "\" picture.")

    # costs = np.squeeze(d["costs"])
    # plt.plot(costs)
    # plt.ylabel('cost')
    # plt.xlabel('iterations (per hundreds)')
    # plt.title("Learning rate =" + str(d["learning_rate"]))
    # plt.show()

    # fname = '/home/yihang/CodeBase/python_try/DeepLearning_byNg/L1W2/datasets/mycat.jpeg'
    # img = Image.open(fname)
    # # 缩放图像
    # my_image = img.resize((1, num_px*num_px*3))
    # my_predicted_image = predict(d["w"], d["b"], my_image)

    # We preprocess the image to fit your algorithm.
    # fname = '/home/yihang/CodeBase/python_try/DeepLearning_byNg/L1W2/datasets/mycat.jpeg'
    fname = '/home/yihang/CodeBase/python_try/DeepLearning_byNg/L1W2/datasets/mydog.jpg'
    image = np.array(plt.imread(fname))

    img = Image.fromarray(image)
    resized_img = img.resize((num_px, num_px))  # 调整图像大小
    image = np.array(resized_img)  # 如果需要，将Pillow图像转换回numpy数组
    print(image.shape)

    my_image = image.reshape((1, num_px * num_px * 3)).T
    print(my_image.shape)

    # 已弃用
    # my_image = scipy.misc.imresize(image, size=(num_px, num_px)).reshape((1, num_px * num_px * 3)).T

    my_predicted_image = predict(d["w"], d["b"], my_image)
    print(my_predicted_image)

    plt.imshow(image)
    plt.show()
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[
        int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.")