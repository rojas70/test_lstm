import numpy as np
import tensorflow as tf

# hyper-parameters
BATCH_SIZE = 64
DATA_DIM = 5

MODEL_SAVED_PATH = './snapshots/model.ckpt'
MODEL_SAVED_DIR = './snapshots'

MAX_ITERATION = 100000
DISPLAY_INTERVAL = MAX_ITERATION // 100
CHECKPOINT_INTERVAL = MAX_ITERATION // 5

## define model here
global_step = tf.Variable(0, name="global_step", trainable=False)
increase_global_step = global_step.assign(global_step + 1)

## build a simple model  
x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, DATA_DIM])
y = tf.placeholder(tf.float32, shape=[BATCH_SIZE, DATA_DIM])

w = tf.get_variable("w", shape=[BATCH_SIZE, DATA_DIM], initializer = tf.zeros_initializer)
wx = w*x

## optimization
loss = tf.reduce_mean((wx - y)**2)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)


## define initializer for global variables
init_op = tf.global_variables_initializer()

## setup one session
sess = tf.Session()

## new a saver for save and load model
saver = tf.train.Saver()


##############################################################################
## After all the previous preparation work has been done, 
## start to run the model via the following statements
##############################################################################
def generate_data():
    x = np.random.randn(BATCH_SIZE, DATA_DIM)
    y = 2*x
    return x, y


## preload the previous saved model or initialize model from scratch
if tf.train.latest_checkpoint(MODEL_SAVED_DIR) is not None:
    saver.restore(sess, tf.train.latest_checkpoint(MODEL_SAVED_DIR))
    print("snapshots path:", tf.train.latest_checkpoint(MODEL_SAVED_DIR))
else:
    sess.run(init_op) ## initialize variables

## get previous iteration
iteration = sess.run(global_step)
print('previous iteration: ', iteration)


while iteration < MAX_ITERATION:
    iteration += 1
    
    
    ## feed data
    _x, _y = generate_data()
    
    ## train model
    fetches = [train_op, increase_global_step, wx, loss]
    feed_dict = {x:_x, y:_y}
    
    _, _, _wx, _loss = sess.run(fetches, feed_dict=feed_dict)


    ## display results
    if (iteration % DISPLAY_INTERVAL) == 0:
        print('\ndisplay iteration', iteration)
        print('x :', _x[0])
        print('y :', _y[0])
        print('wx:', _wx[0])
        print('loss:', _loss)

    # save model at intervals 
    if (iteration % CHECKPOINT_INTERVAL) == 0:
        print('\nsave model iteration', iteration)
        saver.save(sess, MODEL_SAVED_PATH, global_step=iteration)
        print("save model to the directory:".format(iteration, MODEL_SAVED_DIR))
