from asyncore import read
from glob import glob
from turtle import width
import numpy as np
import cv2

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os
from queue import Queue
from scipy.special import softmax

DATA_PATH = "./"
TEST_FILE = "output1_2.txt"

video_tag = "./video1_2.m4v"
labels = [
    "JUMPING",
    "JUMPING_JACKS",
    "BOXING",
    "WAVING_2HANDS",
    "WAVING_1HAND",
    "CLAPPING_HANDS"
]

skeleton_node_num = 18
n_steps = 32 # 32 timesteps per series
n_input = skeleton_node_num * 2
n_classes = 6
n_hidden = 34 # Hidden layer num of features

np.set_printoptions(suppress=True) 

def load_X(X_path):
    file = open(X_path, 'r')
    X_ = np.array(
        [elem for elem in [
            row.split(',') for row in file
        ]], 
        dtype=np.float32
    )
    file.close()
    blocks = int(len(X_) / n_steps)
    
    X_ = np.array(np.split(X_,blocks))

    return X_ 

def get_one_X(file):
    # file is the .txt file which has already been opened to read
    line = file.readlines(1)
    row = line[0] if len(line) > 0 else None
    if row is None:
        X_ = np.array([np.zeros(36)], dtype=np.float)
    else:
        X_ = np.array(
            [row.split(',')], dtype=np.float
        )
    return X_

def LSTM_RNN(_X, _weights, _biases):
    # model architecture based on "guillaume-chevalier" and "aymericdamien" under the MIT license.

    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    _X = tf.reshape(_X, [-1, n_input])   
    # Rectifies Linear Unit activation function used
    _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, n_steps, 0) 

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
#     lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
#     lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
#     lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
#     outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)
    lstm_cell_1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    outputs, states = tf.nn.static_rnn(lstm_cells, _X, dtype=tf.float32)

    # A single output is produced, in style of "many to one" classifier, refer to http://karpathy.github.io/2015/05/21/rnn-effectiveness/ for details
    lstm_last_output = outputs[-1]
    
    # Linear activation
    return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']

def draw_class_score(canvas, scores):
    y_axis = int(height / len(labels)) + 100
    
    for index in range (len(labels)):
        text = labels[index] + ": " + str(f"{scores[index]:.4f}")
        cv2.putText(canvas, text, (100, y_axis), cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 0, 255), 6)
        y_axis += 100   

def draw_skeleton(canvas, key_point):
    key_point = key_point[0]
    for i in range(len(key_point)):
        if i % 2 == 0:
            cv2.circle(canvas, (int(key_point[i]),int(key_point[i+1])), radius=4, color=(255, 0, 0), thickness=-1)  

def main():
    # Graph input/output
    x = tf.placeholder(tf.float32, [None, n_steps, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])

    # Graph weights
    weights = {
        'hidden': tf.Variable(tf.random_normal([n_input, n_hidden]), name="weight_hidden"), # Hidden layer weights
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0), name="weight_out")
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([n_hidden]), name="bias_hidden"),
        'out': tf.Variable(tf.random_normal([n_classes]), name="bias_out")
    }

    pred = LSTM_RNN(x, weights, biases)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, "./models/model.ckpt")  # restore the model
        print("Model restored.")

        cap = cv2.VideoCapture(video_tag)

        global width
        global height
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Video Size: %d x %d" % (width, height))
        
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('./output1_2.avi', fourcc, 30, (width, height))

        frame_queue = Queue(maxsize=32)
        kp_queue = Queue(maxsize=32)
        preds = np.zeros((6,))
        txt_file = open(TEST_FILE, "r")
        while(cap.isOpened()):
            ret, frame = cap.read()
            kp = get_one_X(txt_file)
            print("kp: {}".format(kp))

            frame_queue.put(frame)
            kp_queue.put(kp)
            if frame_queue.full():
                # X_val = load_X(os.path.join(DATA_PATH, TEST_FILE))
                # print("kp_queue: {}".format(list(kp_queue.queue)))
                preds = sess.run([pred], feed_dict={x: np.array(kp_queue.queue).reshape((-1,32,36))})
                # print("len: {}".format(len(preds)))
                preds = softmax(np.array(preds).reshape(6,))
                # print("Sum: {}".format(preds.sum()))
                print("\nPredict label: {}".format(labels[np.argmax(preds)]))
                frame_queue.get()  # remove the earliest frame
                kp_queue.get()

            print(preds)
            draw_class_score(frame, preds)
            draw_skeleton(frame, kp)
            out.write(frame)
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        cap.release()
        out.release()

        cv2.destroyAllWindows()
    
if __name__=="__main__":
    main()