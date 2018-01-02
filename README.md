# Denoisenet

[arXiv:1701.01698](https://arxiv.org/pdf/1701.01687.pdf)
by Remez et el.

## Result
![ex1](https://raw.githubusercontent.com/isVoid/DenoiseNet/master/doc/ex1.png)

![ex2](https://raw.githubusercontent.com/isVoid/DenoiseNet/master/doc/ex2.png)

## Usage

#### Install Requirements
```shell
pip install -r requirements.txt
```

#### Evaluate with pretrained model:
```shell
python3 denoise_eval.py ./data/n000.tif ./data/c000.tif ./Model/denoisenet/denoise_model_I119000.ckpt-119000.meta ./Model/denoisenet/
```

Where `./Model/denoisenet/denoise_model_I119000.ckpt-119000.meta` is tensorflow graph definition, `./Model/denoisenet/` is where checkpoint file locates.

See more with `python3 denoise_eval -h`

## Experiment

Model is trained on house15K dataset for 24 hours on an AWS g2.8xlarge instance. Training is optimized for multi-gpu and achieved 0.8s/it at minibatch size of 16.

![ex1](https://raw.githubusercontent.com/isVoid/DenoiseNet/master/doc/Loss_time_plot.png)

## Train with your own dataset

#### Loading your dataset
2 ways of providing training data is provided:
- In two separate folders, e.g. `Clean/` and `Noisy/`, clean and noisy images are lined up in the same order. For example:
```
  Clean/

  Clean/001.tif

  Clean/002.tif

  Clean/003.tif

  ...

  Noisy/

  Noisy/001.tif

  Noisy/002.tif

  Noisy/003.tif

  ...
```
Where `Clean/001.tif` and `Noisy/001.tif` are groundtruth and corrupted image of the same training example. Run this line to train.
```shell
python3 denoise_train.py -cpr /path/to/Clean/ -npr /path/to/Noisy/
```

- Wrap your training data in a tfrecords file. This is useful when you have an extra large dataset, and cannot fit entire dataset into memory. With the training data lined up as mentioned in the first method, run this line to start converting:
```shell
python3 denoise_input.py /home/ubuntu/data_dev/ --ycbcr
```
`--ycbcr` converts your images into ycbcr format.

Then run this line to train:
```shell
python3 denoise_train.py -df /path/to/dataset.tfrecords
```
