from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
from PIL import Image
from tqdm import tqdm
from MSE_Var import PatchFilter

import argparse
import os
import threading

def sample_patches_2d(img, shape, num_patches, offset_x = 0, offset_y = 0, random_seed = 0):
    """Overriding sklearn.feature_extraction.image.extract_patches_2d

        Extract patches from a mono/tri chromatic image at random, or at a given location.

        Input:
            img: image, either tri-chromatic (:,:,3), or mono-chromatic (:,:,1)
            shape: shape that needs to be extracted into, tuple of two ints (x, y)
            num_patches: number of patches to extract
            offset_x: make offset on x axis
            offset_y: make offset on y axis
            random_seed: seed to extract the image into
        Output:
            patches: an array of patches of shape (num_patches, shape, 3) or (num_patches, shape, 1)

    """

    assert len(img.shape) == 2 or img.shape[2] == 3, "Image dimension mismatch, should be mono or tri chromatic, %s" % str(img.shape)

    # Patch height and width
    sh, sw = shape
    assert type(sh) is int and type(sw) is int, "Error parsing shape"

    # Image height and width
    ih, iw = img.shape[0], img.shape[1]

    # Effective height and width to sample at with offset
    eh = ih - sh
    ew = iw - sw
    hl = np.arange(eh)[0::2].astype(int)
    wl = np.arange(ew)[0::2].astype(int)
    hl += offset_y
    wl += offset_x

    np.random.seed(random_seed)

    x = np.random.choice(wl, num_patches)
    y = np.random.choice(hl, num_patches)
    ats = list(zip(x, y))

    patches = []
    for at in ats:
        patches.append(extract_single_patch_2d_at(img, shape, at))

    patches = np.array(patches)

    return patches

def extract_single_patch_2d_at(img, shape, at):
    """Extract an image patch from certain location of the given image.

        Input:
            img: image, either tri-chromatic(:,:,3) or mono-chromatic(:,:,1)
            shape: shape that needs to be extract into, tuple of two ints (h, w)
            at: Upper left corner of the patch at image coordinate, tuple of two ints (x, y).

        Output:
            patch: of shape (shape, 3) or (shape, 1)

    """

    assert len(img.shape) == 2 or img.shape[2] == 3, "Image dimension mismatch, should be mono or tri chromatic, %s" % str(image.shape)

    h, w = shape
    x, y = at
    assert type(h) is int and type(w) is int, "Error parsing shape"
    assert type(x) is np.int64 and type(y) is np.int64, "Error parsing at %s %s" % (type(x), type(y))

    assert x < img.shape[1] - w, "Exceeds image size x:%d, img.w-w:%d" % (x, img.shape[1] -w)
    assert y < img.shape[0] - h, "Exceeds image size y:%d, img.h-h:%d" % (y, img.shape[0] -h)

    if len(img.shape) == 2:
        patch = img[y:y+h, x:x+w]
    else:
        patch = img[y:y+h, x:x+w, :]

    return patch


def extract_patches_multi(Noisy_List, Clean_List, Noisy_out = './Images/NoisyPatches/', Clean_out = './Images/CleanPatches/',
                            patches_per_image = 10, patch_shape = (128 ,128), offset_x = 0, offset_y = 0):
    """Extract Images into patches (Multi-threaded ver.)

        Input:
            Noisy_List, Clean_List: A list of filenames for noisy and clean images
            Noisy_out, Clean_out: Output directory for noisy and clean patches

        File Output:
            Patches of file written to Noisy_out and Clean_out

        Return:
            None
    """

    assert len(Noisy_List) == len(Clean_List), "Sanity Check: Noisy, Clean images list length mismatch"

    existing_patches_n = get_file_list(Noisy_out)
    existing_patches_c = get_file_list(Clean_out)

    if not len(existing_patches_c) == len(existing_patches_n):
        raise IOError("Existing file count mismatch in output folder, possibility of mismatch of ref.")

    count_base = len(existing_patches_c)
    print ("Output Folder Index Starting from %d" % count_base)

    """Function onto worker thread:
        Input:
            Noisy, Clean: Path for Clean Images
            local_c: local variable in place of count
            local_i: local variable in place of i

        Assumption:
            count_base, Noisy_out, Clean_out, patches_per_image, patch_shape does not
            change through out the extraction.
    """
    def _extract(Noisy, Clean, local_i, random_seed):

        clean_img = np.array(Image.open(Clean))
        noisy_img = np.array(Image.open(Noisy))

        try:
            if not (clean_img.shape[0] == noisy_img.shape[0] and clean_img.shape[1] == noisy_img.shape[1]):
                raise ValueError("Clean(%s) and Noisy(%s) image size mismatch" % (Clean_List[i], Noisy_List[i]))
        except ValueError as e:
            print (e)
            return

        if clean_img.shape[2] == 4:
            # Discard Alpha channel
            clean_img = clean_img[:, :, 0:3]
            noisy_img = noisy_img[:, :, 0:3]

        patches_c = sample_patches_2d(clean_img, patch_shape, patches_per_image, offset_x, offset_y, random_seed)
        patches_n = sample_patches_2d(noisy_img, patch_shape, patches_per_image, offset_x, offset_y, random_seed)

        try:
            if not patches_c.shape[0] == patches_per_image and patches_n.shape[0] == patches_per_image:
                raise ValueError("Extracted Patches number mismatch: clean(%s), noisy(%s), patches_per_image(%s)" % (str(patches_c.shape[0]), str(patches_n.shape[0], str(patches_per_image))))
        except ValueError:
            return

        for n in range(patches_c.shape[0]):
            name_c = "c" + str(count_base + local_i + n).zfill(7) + ".tiff"
            name_n = "n" + str(count_base + local_i + n).zfill(7) + ".tiff"

            Image.fromarray(patches_c[n]).save(Clean_out + name_c)
            Image.fromarray(patches_n[n]).save(Noisy_out + name_n)

    random_state = np.random.randint(0, 10000, 1)
    max_thread = 8

    with tqdm(total = len(Noisy_List), desc = "Extracting Patches", unit = 'frames') as pbar:
        for i in range(len(Noisy_List)):

            # Block for too many threads created, no racing condition here.
            while (threading.active_count() > max_thread):
                pass

            random_state += 1
            threading.Thread(target = _extract, args = (Noisy_List[i], Clean_List[i], i * patches_per_image, random_state)).start()

            pbar.update(1)

    # Hold till all threads finishes.
    while(threading.active_count() > 1):
        pass

    pbar.close()


def extract_patches(Noisy_List, Clean_List, Noisy_out = './Images/NoisyPatches/', Clean_out = './Images/CleanPatches/',
                    patches_per_image = 50, patch_shape = (128 ,128), offset_x = 0, offset_y = 0):
    """Extract Images into patches

        Input:
            Noisy_List, Clean_List: A list of filenames for noisy and clean images
            Noisy_out, Clean_out: Output directory for noisy and clean patches

        File Output:
            Patches of file written to Noisy_out and Clean_out

        Return:
            None
    """

    assert len(Noisy_List) == len(Clean_List), "Sanity Check: Noisy, Clean images list length mismatch"

    existing_patches_n = get_file_list(Noisy_out)
    existing_patches_c = get_file_list(Clean_out)

    if not len(existing_patches_c) == len(existing_patches_n):
        raise IOError("Existing file count mismatch in output folder, possibility of mismatch of ref.")

    count = len(existing_patches_c)
    print ("Output Folder Index Starting from %d" % count)

    random_state = np.random.randint(0, 10000, 1)
    with tqdm(total = len(Noisy_List), desc = "Extracting Patches", unit = 'frames') as pbar:
        for i in range(len(Noisy_List)):
            clean_img = np.array(Image.open(Clean_List[i]))
            noisy_img = np.array(Image.open(Noisy_List[i]))

            try:
                if not (clean_img.shape[0] == noisy_img.shape[0] and clean_img.shape[1] == noisy_img.shape[1]):
                    raise ValueError("Clean(%s) and Noisy(%s) image size mismatch" % (Clean_List[i], Noisy_List[i]))
            except ValueError as e:
                print (e)
                continue

            if clean_img.shape[2] == 4:
                # Discard Alpha channel
                clean_img = clean_img[:, :, 0:3]
                noisy_img = noisy_img[:, :, 0:3]

            patches_c = sample_patches_2d(clean_img, patch_shape, patches_per_image, offset_x, offset_y, random_state)
            patches_n = sample_patches_2d(noisy_img, patch_shape, patches_per_image, offset_x, offset_y, random_state)

            # Empirical values on discard.
            patch_filter = PatchFilter(MSE_thres = 333, Var_thres = 500, total = len(patches_c))
            for j in range(patches_c.shape[0]):
                ci = Image.fromarray(patches_c[j])
                ni = Image.fromarray(patches_n[j])
                # Difference between Noisy and Clean too large.
                if patch_filter.filter_with_mse(patches_c[j], patches_n[j]):
                    continue
                if patch_filter.filter_with_Var(patches_c[j]):
                    continue
                name = "c" + str(count).zfill(7) + ".tiff"
                ci.save(os.path.join(Clean_out, name))
                name = "n" + str(count).zfill(7) + ".tiff"
                ni.save(os.path.join(Noisy_out, name))
                count += 1

            patch_filter.print_discarded_stat()

            random_state += 1
            pbar.update(1)
        pbar.close()


def get_file_list(dir):
    """Get List of files from directory

        Input: DIR, directory to retrieve content from

        Return: filename_list, list of contents filename, sorted by sorted()

    """

    if not os.path.exists(dir):
        raise IOError("%s does not exsit." % dir)

    filename_list = []

    for root, _ , files in os.walk(dir):
        files = sorted(files)
        for f in files:
            filename_list.append(os.path.join(root, f))

    return filename_list


def main(args):

    Noisy_dir = args.Noisy_dir
    Clean_dir = args.Clean_dir
    Noisy_output = args.Noisy_output
    Clean_output = args.Clean_output

    ox = args.offset_x
    oy = args.offset_y

    if not os.path.exists(Noisy_output):
        print ("Creating ", Noisy_output)
        os.mkdir(Noisy_output)

    if not os.path.exists(Clean_output):
        print ("Creating ", Clean_output)
        os.mkdir(Clean_output)

    ppi = args.Patches_per_image
    pw = args.Patch_width

    Noisy_List = get_file_list(Noisy_dir)
    Clean_List = get_file_list(Clean_dir)

    # extract_patches_multi(Noisy_List, Clean_List, Noisy_output, Clean_output, ppi, (pw, pw), ox, oy)
    extract_patches(Noisy_List, Clean_List, Noisy_output, Clean_output, ppi, (pw, pw), ox, oy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Helper function to extract patches for deepdenoisenet")

    parser.add_argument('Noisy_dir', type = str, help = "Directory for your noisy images")
    parser.add_argument('Clean_dir', type = str, help = "Directory for your reference clean images")
    parser.add_argument('--Noisy_output', '-no', type = str, help = "Directory to output noisy patches", default = "./Images/NoisyPatches/")
    parser.add_argument('--Clean_output', '-co', type = str, help = "Directory to output clean patches", default = "./Images/CleanPatches/")
    parser.add_argument('--Patches_per_image', '-ppi', type = int, help = "Patches to extract per image", default = 10)
    parser.add_argument('--Patch_width', '-pw', type = int, help = "Width of patch, which is square", default = 128)
    parser.add_argument('--offset_x', '-ox', type = int, help = "x offset to align mosaic", default = 0)
    parser.add_argument('--offset_y', '-oy', type = int, help = "y offset to align mosaic", default = 0)

    args = parser.parse_args()

    main(args)
