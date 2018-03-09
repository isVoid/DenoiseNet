from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import Denoisenet
import tensorflow as tf

import argparse
from random import shuffle
import time
import os

from tensorflow.python.client import timeline

import denoise_input

PATCH_SHAPE = (128 ,128, 3)
LOGDIR = "./tmp/logs/"

def random_mini_batches(X_train, y_train, mini_batch_size=200, seed = 2):

    """Helper function to extract mini batches from the whole training set.

        Input:
            X_train: Noisy Patches training set
            y_train: Clean Patches training set (long exposures)
            mini_batch_size: Number of patches contained in a mini batch
            seed: We use different seeds each epoch to maintain randomness.

        Output:
            minibatches: python list of tuple (X_batch, y_batch)

    """

    """Number of batches and extra batch that does not fit into a whole batch"""
    num_batches = X_train.shape[0] // mini_batch_size
    res = X_train.shape[0] % mini_batch_size

    """Shuffle entire dataset"""
    sidx = np.arange(X_train.shape[0])
    np.random.shuffle(sidx)
    X = X_train[sidx]
    y = y_train[sidx]

    """Create an index array that refer to the multiples of mini_batch_size"""
    np.random.seed(seed)
    idx = np.arange(X_train.shape[0] - res)
    idx = idx.reshape(-1, mini_batch_size)

    """Create a view that refer X_train and y_train up to multiples of mini_batch_size"""
    X_train_mul = X[0:num_batches * mini_batch_size, :]
    y_train_mul = y[0:num_batches * mini_batch_size, :]

    """Extract the multiple parts into mini_batches"""
    X_train_minis = X_train[idx, :]
    y_train_minis = y_train[idx, :]

    assert X_train_minis.shape == (num_batches, mini_batch_size, PATCH_SHAPE[0], PATCH_SHAPE[1], PATCH_SHAPE[2]), "X shape error: %s" % str(X_train_minis.shape)
    assert y_train_minis.shape == (num_batches, mini_batch_size, PATCH_SHAPE[0], PATCH_SHAPE[1], PATCH_SHAPE[2]), "y shape error: %s" % str(y_train_minis.shape)

    minibatches = []
    for i in range(X_train_minis.shape[0]):
        minibatches.append((X_train_minis[i], y_train_minis[i]))

    if not res == 0:
        """Append the fractional parts"""
        X_train_res = X[num_batches * mini_batch_size:, :]
        y_train_res = y[num_batches * mini_batch_size:, :]

        minibatches.append((X_train_res, y_train_res))
        assert len(minibatches) == num_batches + 1

    return minibatches

def train(X, y, learning_rate = 1e-3, mini_batch_size = 16, debug = False):

    """Function to train deep denoise model

        Input:
            X: numpy array of size (N, PATCH_SHAPE), noisy patches input
            y: numpy array of size X.shape, clean patches input
            learning_rate: learning rate of gradient descent
            mini_batch_size: number of patches of each mini batch
            num_epochs: number of epochs to train on


        Dependencies:
            feed_forward(), where model is defined.

        Console Output:
            epoch_cost: Summed meaned squred error of each epoch

        File Output:
            model: Using tf.Saver() to save trained model.

        Return:
            loss_hist: list of all losses.

    """

    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

    # Build Model
    with tf.device("cpu:0"):
        X_train = tf.placeholder(dtype=tf.float32, shape=(None, 128, 128, 3), name = "X_train")
        y_train = tf.placeholder(dtype=tf.float32, shape=(None, 128, 128, 3), name = "y_train")

    tf.summary.image('input', X_train, 3)
    tf.summary.image('groundtruth', y_train, 3)

    with tf.device("gpu:1"):
        print ("Y Channel Net")
        denoisedY, lossY = Denoisenet.feed_forward(X_train[:, :, :, 0], y_train[:, :, :, 0], scope = "Y")
    with tf.device("gpu:2"):
        print ("Cb Channel Net")
        denoisedCb, lossCb = Denoisenet.feed_forward(X_train[:, :, :, 1], y_train[:, :, :, 1], scope = "Cb")
    with tf.device("gpu:3"):
        print ("Cr Channel Net")
        denoisedCr, lossCr = Denoisenet.feed_forward(X_train[:, :, :, 2], y_train[:, :, :, 2], scope = "Cr")

    with tf.device("gpu:0"):
        denoised = tf.stack([denoisedY[:,:,:,0], denoisedCb[:,:,:,0], denoisedCr[:,:,:,0]], axis = 3, name = "denoised")
        print (denoised.name)
        loss = tf.reduce_sum([lossY, lossCb, lossCr], keep_dims = False)

        tf.summary.scalar("Mean Squared Error", loss)
        tf.summary.image('denoised', denoised, 3)

    Y_vars = tf.trainable_variables(scope = "Y")
    Cb_vars = tf.trainable_variables(scope = "Cb")
    Cr_vars = tf.trainable_variables(scope = "Cr")

    with tf.device("gpu:1"):
        optim_Y = tf.train.AdamOptimizer(learning_rate = learning_rate)
        gvs_Y = optim_Y.compute_gradients(loss, var_list = Y_vars)
        trainop_Y = optim_Y.apply_gradients(gvs_Y)
    with tf.device("gpu:2"):
        optim_Cb = tf.train.AdamOptimizer(learning_rate = learning_rate)
        gvs_Cb = optim_Y.compute_gradients(loss, var_list = Cb_vars)
        trainop_Cb = optim_Cb.apply_gradients(gvs_Cb)
    with tf.device("gpu:3"):
        optim_Cr = tf.train.AdamOptimizer(learning_rate = learning_rate)
        gvs_Cr = optim_Y.compute_gradients(loss, var_list = Cr_vars)
        trainop_Cr = optim_Cr.apply_gradients(gvs_Cr)

    init = tf.global_variables_initializer()

    # Clear previously saved model
    model_dir = "./Model/denoisenet"
    for f in os.listdir(model_dir):
        os.remove(os.path.join(model_dir, f))

    for f in os.listdir(LOGDIR):
        os.remove(os.path.join(LOGDIR, f))

    config = tf.ConfigProto(allow_soft_placement = True)
    with tf.Session(config = config) as sess:

        if debug:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
            run_metadata = tf.RunMetadata()

        sess.run(init)

        saver = tf.train.Saver()
        seed = 0    #Seed to shuffle mini batches

        smooth_loss = 0
        smooth_time = 0
        smooth_alpha = 0.9

        summ = tf.summary.merge_all()
        writer = tf.summary.FileWriter(LOGDIR)
        writer.add_graph(sess.graph)

        tf.Graph().finalize() #No update to graph should be performed in the loop
        i = 0
        while(True):

            cost = 0.

            """Extract into mini batches and shuffle"""
            seed += 1
            minibatches = random_mini_batches(X, y, mini_batch_size, seed)

            for minibatch in minibatches:

                (mini_batch_X, mini_batch_y) = minibatch

                mini_batch_X = mini_batch_X.astype('float32') / 255.
                mini_batch_y = mini_batch_y.astype('float32') / 255.

                mini_batch_count = mini_batch_X.shape[0]

                t = time.time()
                _, _, _, mini_batch_loss, s, d = sess.run([trainop_Y, trainop_Cb, trainop_Cr, loss, summ, denoised], feed_dict = {X_train : mini_batch_X, y_train: mini_batch_y}, options=run_options, run_metadata=run_metadata)
                t = time.time() - t

                cost += mini_batch_loss / mini_batch_count

                writer.add_summary(s, i)

                smooth_loss = (smooth_alpha) * smooth_loss + (1-smooth_alpha) * cost
                smooth_time = (smooth_alpha) * smooth_time + (1-smooth_alpha) * t
                print ("Loss after iterations %d, %f, smooth, %f, time this iter %f, smooth %f" % (i, cost, smooth_loss, t, smooth_time))

                if debug:
                    # Create the Timeline object, and write it to a json
                    tl = timeline.Timeline(run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    with open('./tmp/timeline.json', 'w') as f:
                        f.write(ctf)

                i += 1
                if i % 1000 == 0:
                    saver.save(sess, './Model/denoisenet/denoise_model_I%d.ckpt' % i, global_step = i)


def train_tensors(X, y, learning_rate = 1e-3, mini_batch_size = 16, debug = False):

    """Function to train deep denoise model

        Input:
            X: tensor of size (N, PATCH_SHAPE), noisy patches input
            y: tensor of size X.shape, clean patches input
            learning_rate: learning rate of gradient descent
            mini_batch_size: number of patches of each mini batch
            num_epochs: number of epochs to train on


        Dependencies:
            feed_forward(), where model is defined.

        Console Output:
            epoch_cost: Summed meaned squred error of each epoch

        File Output:
            model: Using tf.Saver() to save trained model.

        Return:
            loss_hist: list of all losses.

    """

    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

    # Build Model
    with tf.device("/gpu:0"):
        tf.summary.image('input', X, 3)
        tf.summary.image('groundtruth', y, 3)

    with tf.device("/gpu:0"):
        print ("Y Channel Net")
        denoisedY, lossY = Denoisenet.feed_forward(X[:, :, :, 0], y[:, :, :, 0], scope = "Y")
    with tf.device("/gpu:1"):
        print ("Cb Channel Net")
        denoisedCb, lossCb = Denoisenet.feed_forward(X[:, :, :, 1], y[:, :, :, 1], scope = "Cb")
    with tf.device("/gpu:2"):
        print ("Cr Channel Net")
        denoisedCr, lossCr = Denoisenet.feed_forward(X[:, :, :, 2], y[:, :, :, 2], scope = "Cr")

    with tf.device("/gpu:0"):
        denoised = tf.stack([denoisedY[:,:,:,0], denoisedCb[:,:,:,0], denoisedCr[:,:,:,0]], axis = 3, name = "denoised")
        print (denoised.name)
        loss = tf.reduce_sum([lossY, lossCb, lossCr], keep_dims = False)

        tf.summary.scalar("Mean Squared Error", loss)
        tf.summary.image("denoised", denoised, 3)


    Y_vars = tf.trainable_variables(scope = "Y")
    Cb_vars = tf.trainable_variables(scope = "Cb")
    Cr_vars = tf.trainable_variables(scope = "Cr")

    with tf.device("/gpu:0"):
        optim_Y = tf.train.AdamOptimizer(learning_rate = learning_rate)
        gvs_Y = optim_Y.compute_gradients(loss, var_list = Y_vars)
        trainop_Y = optim_Y.apply_gradients(gvs_Y)
    with tf.device("/gpu:1"):
        optim_Cb = tf.train.AdamOptimizer(learning_rate = learning_rate)
        gvs_Cb = optim_Y.compute_gradients(loss, var_list = Cb_vars)
        trainop_Cb = optim_Cb.apply_gradients(gvs_Cb)
    with tf.device("/gpu:2"):
        optim_Cr = tf.train.AdamOptimizer(learning_rate = learning_rate)
        gvs_Cr = optim_Y.compute_gradients(loss, var_list = Cr_vars)
        trainop_Cr = optim_Cr.apply_gradients(gvs_Cr)

    init = tf.global_variables_initializer()

    # Clear previously saved model
    model_dir = "./Model/denoisenet"
    for f in os.listdir(model_dir):
        os.remove(os.path.join(model_dir, f))

    for f in os.listdir(LOGDIR):
        os.remove(os.path.join(LOGDIR, f))

    config = tf.ConfigProto(allow_soft_placement = True)
    with tf.Session(config = config) as sess:

        if debug:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
            run_metadata = tf.RunMetadata()

        sess.run(init)

        saver = tf.train.Saver()
        seed = 0    #Seed to shuffle mini batches

        summ = tf.summary.merge_all()
        writer = tf.summary.FileWriter(LOGDIR)
        writer.add_graph(sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        i = 0

        smooth_loss = 1.
        smooth_time = 1.
        smooth_alpha = 0.9

        tf.Graph().finalize() #No update to graph should be performed in the loop
        try:
            while not coord.should_stop():

                cost = 0.

                t = time.time()
                _, _, _, mini_batch_loss, s = sess.run([trainop_Y, trainop_Cb, trainop_Cr, loss, summ], options=run_options, run_metadata=run_metadata)
                t = time.time() - t

                cost += mini_batch_loss / mini_batch_size

                writer.add_summary(s, i)

                i += 1
                smooth_loss = (smooth_alpha) * smooth_loss + (1-smooth_alpha) * cost
                smooth_time = (smooth_alpha) * smooth_time + (1-smooth_alpha) * t
                print ("Loss after iterations %d, %f, smooth %f, time this iter %f, smooth %f" % (i, cost, smooth_loss, t, smooth_time))

                if debug:
                    # Create the Timeline object, and write it to a json
                    tl = timeline.Timeline(run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    with open('./tmp/timeline.json', 'w') as f:
                        f.write(ctf)

                if i % 1000 == 0:
                    saver.save(sess, './Model/denoisenet/denoise_model_I%d.ckpt' % i, global_step = i)

        except tf.errors.OutOfRangeError:
            print('Finished one epoch, %d steps' % i)
        finally:
            coord.request_stop()

        coord.join(threads)
        sess.close()


def main(args):

    lr = args.learning_rate if args.learning_rate else 1e-3
    mbs = args.mini_batch_size if args.mini_batch_size else 16
    print ("Training with learning rate %E, batch size %d" % (lr, mbs))

    if args.dataset_file:
        dataset_file = args.dataset_file

        cims, nims = denoise_input.input_tfRecords(dataset_file, mbs)

        train_tensors(nims, cims, learning_rate = lr)

    else:
        clean_patches_root = args.clean_patches_root
        noisy_patches_root = args.noisy_patches_root

        clean_patches, noisy_patches = denoise_input.load_patches(CleanRoot = clean_patches_root, NoisyRoot = noisy_patches_root)
        train(X = noisy_patches, y = clean_patches, learning_rate = lr, mini_batch_size = mbs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Trainer for deep denoise net.")
    parser.add_argument('--clean_patches_root', '-cpr', type=str, help="Path to the folder of clean patches", default="./Images/CleanPatches/")
    parser.add_argument('--noisy_patches_root', '-npr', type=str, help="Path to the folder of noisy pathces", default="./Images/NoisyPatches/")
    parser.add_argument('--dataset_file', '-df', type=str, help="Path to tfrecord file, if defined, cpr and npr will be ignored.")
    parser.add_argument('--learning_rate', '-lr', type=float, help="Learning Rate of train")
    parser.add_argument('--mini_batch_size', '-mbs', type=int, help="Mini batch size")

    args = parser.parse_args()

    os.makedirs("./tmp/logs/", exist_ok=True)

    main(args)
