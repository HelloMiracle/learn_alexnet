import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
learningRate=0.001
trainintIter=200000
batchSize=128
displayStep=10
inputSize=784
classSize=10
keepProb=0.75

x=tf.placeholder(tf.float32,[None,inputSize])
y=tf.placeholder(tf.float32,[None,classSize])
dropout=tf.placeholder(tf.float32)

def layer_bn(x,isTrain,name="bnormalization",moving_decay=0.9,eps=1e-5):
    x_shape=x.shape
    paramShape=x_shape[-1]
    with tf.variable_scope(name):
        gamma = tf.get_variable('gamma', paramShape, initializer=tf.constant_initializer(1))
        beta = tf.get_variable('beta', paramShape, initializer=tf.constant_initializer(0))
        axes=list(range(len(x_shape)-1))
        batch_mean,batch_var=tf.nn.moments(x,axes,name='moments')
        ema=tf.train.ExponentialMovingAverage(moving_decay)
        def mean_var__updata():
            ema_apply_op=ema.apply([batch_mean,batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
    mean,var=tf.cond(tf.equal(isTrain,True),mean_var__updata,lambda:(ema.average(batch_mean),ema.average(batch_var)))
    return tf.nn.batch_normalization(x,mean,var,beta,gamma,eps)


def conv2d(name,x,w,b,strides=1):
    x=tf.nn.conv2d(x,w,strides=[1,strides,strides,1],padding="SAME")
    x=tf.nn.bias_add(x,b)
    bn_x = layer_bn(x, True, name=name)
    return tf.nn.relu(bn_x)

def max_pool(x,k=2):
    return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding="SAME")

def norm(name,l_input,lsize=4):
    return tf.nn.lrn(l_input,lsize,bias=1.0,alpha=0.001/9.0,beta=0.75,name=name)

'''
weights={
    'wc1':tf.Variable(tf.random_normal([5,5,1,96])),
    'wc2':tf.Variable(tf.random_normal([5,5,96,256])),
    'wc3':tf.Variable(tf.random_normal([3,3,256,384])),
    'wc4':tf.Variable(tf.random_normal([3,3,384,384])),
    'wc5':tf.Variable(tf.random_normal([3,3,384,256])),
    'wd1':tf.Variable(tf.random_normal([4*4*256,4096])),
    'wd2':tf.Variable(tf.random_normal([4096,4096])),
    'out':tf.Variable(tf.random_normal([4096,10])),
}

bias={
    'bc1':tf.Variable(tf.random_normal([96])),
    'bc2': tf.Variable(tf.random_normal([256])),
    'bc3': tf.Variable(tf.random_normal([384])),
    'bc4': tf.Variable(tf.random_normal([384])),
    'bc5': tf.Variable(tf.random_normal([256])),
    'bd1': tf.Variable(tf.random_normal([4096])),
    'bd2': tf.Variable(tf.random_normal([4096])),
    'out': tf.Variable(tf.random_normal([classSize]))
}
'''
weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64])),
    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    'wd1': tf.Variable(tf.random_normal([4*4*256, 1024])),
    'wd2': tf.Variable(tf.random_normal([1024, 1024])),
    'out': tf.Variable(tf.random_normal([1024, 10]))
}
bias = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([128])),
    'bc3': tf.Variable(tf.random_normal([256])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'bd2': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([classSize]))
}
def build_alexnet(x,weight,bias,dropout):
    x=tf.reshape(x,shape=[-1,28,28,1])

    conv1=conv2d('conv1',x,weight['wc1'],bias['bc1'])
    pool1=max_pool(conv1,k=2)
   # norm1=norm('norm1',pool1,lsize=4)
    #bn1=layer_bn(pool1,True,'bn1')
    conv2 = conv2d('conv2',pool1, weight['wc2'], bias['bc2'])
    pool2 = max_pool(conv2, k=2)
   # norm2 = norm('norm2', pool2, lsize=4)
    #bn2 = layer_bn(pool2, True, 'bn1')
    conv3 = conv2d('conv3',pool2, weight['wc3'], bias['bc3'])
    pool3 = max_pool(conv3, k=2)
   # norm3 = norm('norm3', pool3, lsize=4)

   # conv4 = conv2d('conv4', norm3, weight['wc4'], bias['bc4'])


    #conv5 = conv2d('conv5',conv4, weight['wc5'], bias['bc5'])
   # pool5 = max_pool(conv5, k=2)
   # norm5 = norm('norm5', pool5, lsize=4)

    fc1=tf.reshape(pool3,shape=[-1,weight['wd1'].get_shape().as_list()[0]])
    fc1=tf.add(tf.matmul(fc1,weight['wd1']),bias['bd1'])
    fc1=tf.nn.relu(fc1)

    fc1=tf.nn.dropout(fc1,dropout)

    fc2 = tf.reshape(fc1, shape=[-1, weight['wd2'].get_shape().as_list()[0]])
    fc2 = tf.add(tf.matmul(fc2, weight['wd2']), bias['bd2'])
    fc2 = tf.nn.relu(fc2)

    fc2 = tf.nn.dropout(fc2, dropout)

    out=tf.add(tf.matmul(fc2, weight['out']), bias['out'])

    return out


pred_y=build_alexnet(x,weights,bias,dropout)

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_y,labels=y))
optimizer=tf.train.AdadeltaOptimizer(learning_rate=learningRate).minimize(cost)
correct_pred=tf.equal(tf.argmax(pred_y,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    trainStep=1
    while trainStep*batchSize<trainintIter:
        batch_x,batch_y=mnist.train.next_batch(batchSize)
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y,dropout:keepProb})
        if trainStep%displayStep==0:
            loss,acc=sess.run([cost,accuracy],feed_dict={x:batch_x,y:batch_y,dropout:1.0})
            print("iter"+str(trainStep*batchSize)+" minibatchLoss="+"{:.6f}".format(loss)+" train Accuracy="+"{:.5f}".format(acc))
        trainStep+=1
    print("finish train")

    print("testdata accuracy",sess.run(accuracy,feed_dict={x:mnist.test.image[:256],y:mnist.test.image[:256],dropout:1.0}))