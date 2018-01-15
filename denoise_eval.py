from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import Denoisenet
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

import time
import os

import argparse

PATCH_SHAPE = (128 ,128, 3)

def _log10(x):
    return np.log(x) / np.log(10)

def psnr(test, gt):
    MSE = np.mean((test - gt) ** 2)
    MAX = np.max(gt)
    _psnr = 20 * _log10(MAX) - 10 * _log10(MSE)
    return _psnr

def image_to_patches(img_pad, stride_x, stride_y, mini_batch_size = 200):
    """Helper function to convert an image input to patches
        Input:
            img_pad: image of size (Hp, Wp, 3), numpy array, padded to multiples of PATCH_SHAPE
        Output:
            batches: batches of patches of shape (mini_batch_size, PATCH_SHAPE)
            res: valid patches output in the last batch.
    """
    Hp, Wp, C = img_pad.shape
    batches = None

    num_patches_x = (Wp - PATCH_SHAPE[1]) / stride_x + 1
    num_patches_y = (Hp - PATCH_SHAPE[0]) / stride_y + 1

    total_patches = int(num_patches_x * num_patches_y)
    res = int(total_patches % mini_batch_size)

    # print ("num_patches_x", num_patches_x)
    # print ("num_patches_y", num_patches_y)
    # print ("total patches", total_patches)
    # print ("mini_batch_size", mini_batch_size)
    # print ("res", res)
    # print ("n_patches", np.ceil(float(total_patches) / mini_batch_size))

    if not res == 0:
        n_patches = total_patches + (mini_batch_size - res)
    else:
        n_patches = total_patches

    batches = np.zeros((n_patches, *PATCH_SHAPE))

    x = y = 0
    for n in range(total_patches):
        # Extract patches from image
        patch = np.array([img_pad[y:y+PATCH_SHAPE[1], x:x+PATCH_SHAPE[0], :]])
        assert patch.shape == (1, *PATCH_SHAPE), "Shape mismatch, %s" % str(patch.shape)

        # batches shape: (N, H, W, C)
        batches[n, :, :, :] = patch

        # Next patch along X
        if (x + PATCH_SHAPE[1] < Wp):
            x += stride_x
        # New line, start from X=0
        elif x + PATCH_SHAPE[1] >= Wp and y + PATCH_SHAPE[0] < Hp:
            y += stride_y
            x = 0
        # Last patch
        elif x + PATCH_SHAPE[1] >= Wp and y + PATCH_SHAPE[0] >= Hp:
            break

    # print ("batch shape before", batches.shape)
    # Network takes fixed sized input (mini_batch, PATCH_SHAPE), append zeros to meet shape convention.
    if not res == 0:
        for i in range(mini_batch_size-res):
            batches = np.concatenate((batches, patch), axis=0)

    # print ("batch shape after", batches.shape)

    batches = batches.reshape(-1, mini_batch_size, PATCH_SHAPE[0], PATCH_SHAPE[1], PATCH_SHAPE[2])

    # print ("batch shape after", batches.shape)
    # print ("total batches", batches.shape[0])

    return batches, res

def patches_to_image(img, batches_of_patches, stride_x, stride_y):

    """Takes an input of batches of patches, restore them back to original image
        Input:
        img: input image, used to define output shape (H, W, C)
        batches_of_patches, of shape (mini_batch_size, PATCH_SHAPE)
        stride_x: the stride of the patches on x axis
        stride_y: the stride of the patches on y axis
        Output:
        image reconstructed from patches, with shape (H, W, C)
    """
    H, W, C = img.shape
    output = np.zeros(img.shape)

    crop_in_x = (int)((PATCH_SHAPE[1] - stride_x) / 2)
    crop_in_y = (int)((PATCH_SHAPE[0] - stride_y) / 2)

    h,w,c = PATCH_SHAPE
    x = y = 0

    j = 0
    for patches in batches_of_patches:
        for i in range(patches.shape[0]):
            # Get patch
            p = patches[i, crop_in_y:h-crop_in_y, crop_in_x:w-crop_in_x, :]

            pt = (p * 255).astype("uint8")
            j += 1
            # Stitching
            output[y + crop_in_y : y + PATCH_SHAPE[0] - crop_in_y, x + crop_in_x : x + PATCH_SHAPE[1] - crop_in_x, :] = p

            # Next patch along X
            if x + PATCH_SHAPE[1] < W:
                x += stride_x
            # New line with X=0
            elif x + PATCH_SHAPE[1] >= W and y + PATCH_SHAPE[0] < H:
                y += stride_y
                x = 0
            # Last patch
            elif x + PATCH_SHAPE[1] >= W and y + PATCH_SHAPE[0] >= H:
                break

    return output


def eval_patch(X, y, sess = None):
    """Evaluate some patches input using trained model
        Inputs:
            X: Noisy Image patches, size (N, PATCH_SHAPE), numpy array
            y: Ground truth image patches, size (N, PATCH_SHAPE), numpy array
            sess: session with model preloaded, passed in when denoising whole image.
        Return:
            output: evaluated output from the network.
            loss: meaned squared error between output and ground truth
    """

    assert sess is not None, "Session is NoneType when evaluating single patch. Check eval_image()."

    if not X.shape == y.shape:
        raise ValueError("Shape mismatch when evaluating single patch, shape X %s, shape Y %s" % (str(X.shape), str(y.shape)))

    graph = tf.get_default_graph()

    X_train = graph.get_tensor_by_name("input/mul_1:0")
    y_train = graph.get_tensor_by_name("input/mul:0")
    denoised = graph.get_tensor_by_name("denoised:0")
    loss = Denoisenet.loss(denoised, y_train)

    denoised, loss = sess.run([denoised, loss], feed_dict={X_train : X, y_train : y})

    return denoised, loss

def eval_image(X, y, model = None, checkpoint = None, mini_batch_size = 16, crop_in = 5):

    """Evaluate a full image using a trained model
        Achieved by dividing image into various patches and apply model individually.
        Inputs:
            X: Noisy Image Input, of size (H, W, 3), numpy array.
            y: Ground truth Image (Long Exposure Image), of same size as X, numpy array.
            checkpoint: Tensorflow checkpoint state object path
        Console Outputs:
            Loss: Summed loss of all patches across entire image
        Returns:
            Output: denoised image, of size (H, W, 3), numpy array.
            Total_loss: Aggregated loss over entire image.
    """

    assert model is not None, "Eval Image: Trained model location is not specified."
    assert checkpoint is not None, "Eval Image: Tensorflow Checkpoint is not specified."
    assert X.shape == y.shape, "Eval Image: X(%s) and y(%s) shape mismatch." % (X.shape, y.shape)

    tf.reset_default_graph()

    config = tf.ConfigProto(allow_soft_placement = True)
    with tf.Session(config = config) as sess:
        # Load previous model
        saver = tf.train.import_meta_graph(model)
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint))

        # Get Input shape
        H, W, C = X.shape

        if (C == 1):
            raise ValueError("Monochromatic image not supported.")

        # Evaluate a large image in patches, stride is length - 2*crop
        # To escape convolution artifacts
        stride_x = PATCH_SHAPE[0] - 2 * crop_in
        stride_y = PATCH_SHAPE[1] - 2 * crop_in

        # Pad
        pad_right = stride_x - ((W - PATCH_SHAPE[1]) % stride_x)
        pad_bottom = stride_y - ((H - PATCH_SHAPE[0]) % stride_y)

        # Zero Pad Input to multiples of PATCH_SHAPE
        X_pad = np.pad(X, ((0, pad_bottom), (0, pad_right), (0, 0)), 'constant', constant_values = 0)
        y_pad = np.pad(y, ((0, pad_bottom), (0, pad_right), (0, 0)), 'constant', constant_values = 0)

        Hp, Wp, Cp = X_pad.shape

        x = y = 0
        while (x + PATCH_SHAPE[1] < Wp):
            x += stride_x
        assert x + PATCH_SHAPE[1] == Wp, "Padding on W is wrong."

        while (y + PATCH_SHAPE[0] < Hp):
            y += stride_y
        assert y + PATCH_SHAPE[0] == Hp, "Padding along H is wrong"

        batches_X, resX = image_to_patches(X_pad, stride_x, stride_y, mini_batch_size = mini_batch_size)
        batches_Y, resY = image_to_patches(y_pad, stride_x, stride_y, mini_batch_size = mini_batch_size)

        assert batches_X.shape[0] == batches_Y.shape[0], "Batch num mismatch, X:%d, y:%d" % (batches_X.shape[0], batches_Y.shape[0])
        assert resX == resY, "Residuals mismatch, X%d, y:%d" % (resX, resY)

        total_batch_num = mini_batch_size * (batches_X.shape[0] - 1) + resX

        total_loss = 0.
        output = np.zeros(y_pad.shape)

        output_patches = []
        for i in tqdm(range(batches_X.shape[0]), desc = "Denoise with network"):
            batch_X = batches_X[i]
            batch_y = batches_Y[i]
            denoised_batch, loss = eval_patch(batch_X, batch_y, sess)
            output_patches.append(denoised_batch)
            total_loss += np.sum(loss)

        print ("Average MSE across image: %.6E" % (total_loss / total_batch_num))

        output = patches_to_image(y_pad, output_patches, stride_x, stride_y)

        """Crop to original size"""
        output = output[0:X.shape[0], 0:X.shape[1], 0:3]

        sess.close()
        return output, total_loss

def main(args):
    t = time.time()

    X_path = args.EvalX
    y_path = args.EvalY
    model = args.Model
    ckpt = args.Checkpoint

    crop_in = 8

    if args.Output:
        Output_path = args.Output
    else:
        Output_path = './Output/'

    # Read Images, discard alpha channel
    X = np.asarray(Image.open(X_path).convert("YCbCr"))[:, :, 0:3]
    y = np.asarray(Image.open(y_path).convert("YCbCr"))[:, :, 0:3]

    # Normalize:
    X = X / 255.
    y = y / 255.

    # Process with eval_image
    output, _ = eval_image(X, y, model, ckpt)

    # Write output as image
    name = X_path.split("/")
    name = name[len(name) - 1]

    # Output loses a few crop_in pixels
    _output = output[crop_in:output.shape[0]-crop_in, crop_in:output.shape[1]-crop_in, :]
    _y = y[crop_in:y.shape[0]-crop_in, crop_in:y.shape[1]-crop_in:, :]
    print ("PSNR: ", psnr(_output, _y))

    assert output.shape == X.shape, "Evaluated Output shape(%s) mismatch with Input shape(%s)." % (output.shape, X.shape)

    output = output * 255.

    output = output.astype('uint8')

    # Write output as image
    name = X_path.split("/")
    name = name[len(name) - 1]

    if not os.path.exists(Output_path):
        os.mkdir(Output_path)
    Image.fromarray(output, mode = "YCbCr").convert("RGB").save(Output_path + name)

    t = time.time() - t
    print ("time used(s)", t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Evaluate a single noisy image and compute total loss.")
    parser.add_argument('EvalX', type=str, help="Path of the noisy image to evaluate.")
    parser.add_argument('EvalY', type=str, help="Path of the ground truth image")
    parser.add_argument('Model', type=str, help="Path of the trained Tensorflow Model, this builds tensorflow graph.")
    parser.add_argument('Checkpoint', type=str, help="Path of Tensorflow checkpoint, this restores parameters.")
    parser.add_argument('--Output', type=str, help="Path of the output image")

    args = parser.parse_args()
main(args)
