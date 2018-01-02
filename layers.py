def conv_layer_with_pooling(input, in_channel, out_channel, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5, 5, in_channel, out_channel], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape = [out_channel]), name = "B")
        conv = tf.nn.conv2d(input, w, strides = [1, 1, 1, 1], padding = "SAME")
        act = tf.nn.relu(conv + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return tf.nn.max_pool(act, ksize = [1, 2, 2, 1], strides = [1, 1, 1, 1], padding = "SAME")

def conv_layer_with_no_pooling(input, in_channel, out_channel, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5, 5, in_channel, out_channel], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape = [out_channel]), name = "B")
        conv = tf.nn.conv2d(input, w, strides = [1, 1, 1, 1], padding = "SAME")
        act = tf.nn.relu(conv + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act

def conv_layer_with_sigmoid_act(input, in_channel, out_channel, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5, 5, in_channel, out_channel], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape = [out_channel]), name = "B")
        conv = tf.nn.conv2d(input, w, strides = [1, 1, 1, 1], padding = "SAME")
        act = tf.sigmoid(conv + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        return act

def conv_layer_with_idendity_act(input, in_channel, out_channel, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5, 5, in_channel, out_channel], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape = [out_channel]), name = "B")
        conv = tf.nn.conv2d(input, w, strides = [1, 1, 1, 1], padding = "SAME")
        biased = conv + b
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        return biased
