from PIL import Image
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import os
import re
import argparse

PATCH_SHAPE = (128 ,128, 3)

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def load_patches_as_array(root, TO_YCbCr = True):

    """Helper function to load patches, normalize and convert into numpy arrays

        Input:
            root: folder path for image patches, with file name in sequence.

        Output:
            image_array: A numpy array of size (batch_size, 128, 128, 3)

    """

    if not os.path.exists(root):
        raise IOError(root + " does not exists.")

    image_array = []

    for r, fo, f in os.walk(root):

        f = sorted(f)

        with tqdm(total = len(f), desc = "loading " + root) as pbar:
            for im in f:
                try:
                    im = Image.open(r + im)
                    if TO_YCbCr:
                        im = im.convert('YCbCr')
                except IOError:
                    print ("File Not Found, ", r + im)
                    break
                im = np.asarray(im)
                if im.shape[2] == 4:
                    im = im[:,:,0:3]
                if not im.shape == PATCH_SHAPE:
                    raise ValueError("Wrong Image Patch Size, %s" % str(im.shape))
                image_array.append(im)
                pbar.update(1)

    image_array = np.asarray(image_array)

    return image_array

def load_patches(CleanRoot = "./Images/CleanPatches/", NoisyRoot = "./Images/NoisyPatches/"):

    """Helper function to load Clean and Noisy patches for training.

        Input:
            CleanRoot: Root folder path for Clean patches.
            NoisyRoot: Root folder path for Noisy patches.

        Return:
            clean_patches: size of (N, PATCH_SHAPE), numpy array
            noisy_patches: size of (N, PATCH_SHAPE), numpy array

    """

    assert type(CleanRoot) is str, "Invalid root for clean patches"
    assert type(NoisyRoot) is str, "Invalid root for noisy patches"

    clean_patches = load_patches_as_array(CleanRoot)
    noisy_patches = load_patches_as_array(NoisyRoot)

    """Sanity Check: clean_patches shape same noisy_patches """
    assert clean_patches.shape == noisy_patches.shape, "Shape mismatch of clean_patches(%s) and noisy_patches(%s)" % (str(clean_patches.shape), str(noisy_patches.shape))

    return clean_patches, noisy_patches

def convert_patches_to_tfRecords(root, TO_YCbCr = True, output = None):

    """Helper function to convert a folder of patch files into tf records

        Input:
            root: root folder for Clean and Noisy patches
            TO_YCbCr: convert patches into YCbCr space?

        Return:
            count: number of patches converted

    """

    if output == None:
        dataset_filename = os.path.join(root, 'dataset.tfrecords')
    else:
        dataset_filename = os.path.join(output, 'dataset.tfrecords')

    CleanRoot = os.path.join(root, "Clean")
    NoisyRoot = os.path.join(root, "Noisy")

    if not os.path.exists(CleanRoot) or not os.path.exists(NoisyRoot):
        raise IOError("%s or %s does not exists." % (CleanRoot, NoisyRoot))

    CleanFs = os.listdir(CleanRoot)
    NoisyFs = os.listdir(NoisyRoot)

    assert len(CleanFs) == len(NoisyFs), "File # mismatch"

    CleanFs = sorted(CleanFs)
    NoisyFs = sorted(NoisyFs)

    total_count = len(CleanFs)

    print ('Writing', dataset_filename)
    pbar = tqdm(total = total_count, desc = "loading")

    writer = tf.python_io.TFRecordWriter(dataset_filename)
    for i in range(total_count):

        CleanF = CleanFs[i]
        NoisyF = NoisyFs[i]

        try:
            cim = Image.open(os.path.join(CleanRoot, CleanF))
            nim = Image.open(os.path.join(NoisyRoot, NoisyF))
            if TO_YCbCr:
                cim = cim.convert('YCbCr')
                nim = nim.convert('YCbCr')
        except IOError as e:
            print (e)
            continue

        cim = np.asarray(cim)
        nim = np.asarray(nim)

        if cim.shape[2] == 4:
            cim = cim[:,:,0:3]
        if nim.shape[2] == 4:
            nim = nim[:,:,0:3]

        if not nim.shape == PATCH_SHAPE or not cim.shape == PATCH_SHAPE:
            raise ValueError("Wrong Image Patch Size, %s, %s" %
                    (str(nim.shape), str(cim.shape)))

        # Convert to float64 will cause filesize increase by 8 times.
        # cim = cim / 255.
        # nim = nim / 255.

        cim_raw = cim.tostring()
        nim_raw = nim.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(PATCH_SHAPE[0]),
            'width': _int64_feature(PATCH_SHAPE[1]),
            'depth': _int64_feature(PATCH_SHAPE[2]),
            'clean_raw': _bytes_feature(cim_raw),
            'noisy_raw': _bytes_feature(nim_raw)
        }))

        writer.write(example.SerializeToString())

        pbar.update(1)

    writer.close

    return total_count

def read_tfRecords(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
        features={
            'clean_raw': tf.FixedLenFeature([], tf.string),
            'noisy_raw': tf.FixedLenFeature([], tf.string)
        })

    cim = tf.decode_raw(features['clean_raw'], tf.uint8)
    nim = tf.decode_raw(features['noisy_raw'], tf.uint8)

    cim.set_shape([np.prod(list(PATCH_SHAPE))])
    nim.set_shape([np.prod(list(PATCH_SHAPE))])

    cim = tf.reshape(cim, [*PATCH_SHAPE])
    nim = tf.reshape(nim, [*PATCH_SHAPE])

    # Floatarize before batching expands total memory usage
    # by 8 times!!!
    # cim = tf.cast(cim, tf.float32) * (1. / 255)
    # nim = tf.cast(nim, tf.float32) * (1. / 255)

    # print ("cim", cim)

    return cim, nim

def input_tfRecords(fn, batch_size):
    with tf.name_scope('input'):
        with tf.device('cpu:0'):
            filename_queue = tf.train.string_input_producer(
                [fn], num_epochs = None #train Forever
            )
            cim, nim = read_tfRecords(filename_queue)

            min_after_dequeue = 100000
            cims, nims = tf.train.shuffle_batch(
                [cim, nim], batch_size=batch_size, num_threads = 2,
                capacity = min_after_dequeue + 3 * batch_size,
                min_after_dequeue = min_after_dequeue
            )

        cims = tf.cast(cims, tf.float32) * (1. / 255)
        nims = tf.cast(nims, tf.float32) * (1. / 255)

        print ('cims', cims.name)
        print ('nims', nims.name)

        return cims, nims

def extract_patches_from_image_tensor(clean_imgt, noisy_imgt):
    """
    Extract patches from the given image tensor:
    
    """

def main(args):
    root = args.root
    ycbcr = args.ycbcr if args.ycbcr else True
    output = args.output if args.output else None

    convert_patches_to_tfRecords(root, ycbcr, output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('root', type=str, help = 'root folder for clean and noisy patches')
    parser.add_argument('--ycbcr', action = 'store_true', help = 'preprocess patches into ycbcr space')
    parser.add_argument('--output', type=str, help = '(Optional) Output folder')

    args = parser.parse_args()

    main(args)
