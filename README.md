# Conditional CoopNets

This repository contains a TensorFlow implementation for TPAMI paper "[Cooperative Training of Fast Thinking Initializer and Slow Thinking Solver for
Conditional Learning](http://www.stat.ucla.edu/~jxie/CCoopNets/CCoopNets_file/doc/CCoopNets.pdf)"

## Set Up Environment
We have provided the environment file cCoopNets.yml for setting up the environment. The environment can be set up with one command using conda

```bash
conda env create -f cCoopNets.yml
conda activate cCoopNets
```

## Reference
    @article{DG,
        author = {Jianwen Xie, Zilong Zheng, Xiaolin Fang, Song-Chun Zhu, Ying Nian Wu},
        title = {Cooperative Training of Fast Thinking Initializer and Slow Thinking Solver for Conditional Learning},
        journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
        year = {2021}
    }

## Training

### Conditional image generation on grayscale images

(a) MNIST dataset

    $ python main_mnist.py --category MNIST
    
(b) fashion-MNIST dataset

    $ python main_mnist.py --category fashion_MNIST

    
