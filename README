# TailorGAN

TailorGAN: Making User-Defined Fashion Designs.

  - paper link: https://arxiv.org/abs/2001.06427

### Prerequisites
* Linux system or MacOS
* Python 3.6
* NVIDIA GPU + CUDA

### Installation

* Install PyTorch and dependencies from http://pytorch.org

### Test

* Default, using test.py file to synthesize the garment from the images in the example folder. Pre-trained checkpoints download needed and as the instruction in the checkpoints folder to download.
```sh
$ python test.py
```
* Sample results are presented in the example folder.

### Train

* Training process is divided into two parts, the first part is reconstruction, using following command to train first step, and putting the checkpoints to corresponding folder to train second step.
```sh
$ python collarRecon.py --step step1
```

* Second step, using following command:
```sh
$ python collarSyn.py --step step2
```

### Citation

* If you think our work is useful, please use the following:
```sh
@misc{chen2020tailorgan,
    title={TailorGAN: Making User-Defined Fashion Designs},
    author={Lele Chen and Justin Tian and Guo Li and Cheng-Haw Wu and Erh-Kan King and Kuan-Ting Chen and Shao-Hang Hsieh and Chenliang Xu},
    year={2020},
    eprint={2001.06427},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
