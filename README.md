# Global Deconvolutional Networks for Semantic Segmentation Code

## Installation

You need to install Caffe with the support of Python layer enabled. The Caffe version provided here can differ significantly from the most recent one.
You will also need to download the corresponding `caffemodel` file: [link](https://www.dropbox.com/s/v1bhpprp6unkzd0/gdn.caffemodel?dl=0), and put it into the `model/` folder. 

A running example can be seen in the `gdn_caffe` notebook file. 


## Description
Online segmentation demo is available at http://saildemo.unist.ac.kr/semantic_segmentation/

For details on the underlying model please refer to the following paper:

    @inproceedings{BMVCNekrasovJC16,
      author    = {Vladimir Nekrasov and
                   Janghoon Ju and
                   Jaesik Choi},
      title     = {Global Deconvolutional Networks for Semantic Segmentation},
      booktitle = {Proceedings of the British Machine Vision Conference 2016, {BMVC}
                   2016, York, UK, September 19-22, 2016},
      year      = {2016}
    }
