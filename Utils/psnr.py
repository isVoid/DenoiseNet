from skimage.measure import compare_psnr
import imageio
import os
import numpy as np

croot = "/home/michael/workspace/deep_denoise/Images/Clean/"
nroot = "/home/michael/workspace/deep_denoise/Images/Noisy/"

cs = sorted(os.listdir(croot))
ns = sorted(os.listdir(nroot))

fullcs = [os.path.join(croot, f) for f in cs]
fullns = [os.path.join(nroot, f) for f in ns]

psnrs = []
for i in range(len(fullcs)):
    print (fullcs[i], fullns[i])
    gti = imageio.imread(fullcs[i])
    ti = imageio.imread(fullns[i])

    psnr = compare_psnr(gti, ti)
    if not psnr == float('inf'):
        print (psnr)
        psnrs.append(psnr)
    else:
        print ("Infinite psnr, probably unifrom")

print ("total average psnr: ", np.mean(psnrs))
