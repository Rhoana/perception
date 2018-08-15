# Evaluating 'Graphical Perception' with CNNs

![Image](http://danielhaehn.com/papers/haehn2018evaluating.png) 

**[PAPER](http://danielhaehn.com/papers/haehn2018evaluating.pdf)** | **[SUPPLEMENTAL](http://danielhaehn.com/papers/haehn2018evaluating_supplemental.pdf)** | **[FASTFORWARD](https://vimeo.com/285106317)** | **[VIDEO](https://vimeo.com/280506639)**
  

**Convolutional neural networks can successfully perform many computer vision tasks on images. For visualization, how do CNNs perform when applied to graphical perception tasks? We investigate this question by reproducing Cleveland and McGill’s seminal 1984 experiments, which measured human perception efficiency of different visual encodings and defined elementary perceptual tasks for visualization. We measure the graphical perceptual capabilities of four network architectures on five different visualization tasks and compare to existing and new human performance baselines. While under limited circumstances CNNs are able to meet or outperform human task performance, we find that CNNs are not currently a good model for human graphical perception. We present the results of these experiments to foster the understanding of how CNNs succeed and fail when applied to data visualizations.**

Note: This paper will be presented at [IEEE Vis 2018](http://ieeevis.org/) in Berlin!

## Data

<img src='https://dataverse.harvard.edu/logos/navbar/logo.png' width=150>

The data including trained models, experiments, and results are available as part of the Harvard Dataverse.

Access it [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/7HFT8D)!

## Code

<img src='https://assets-cdn.github.com/images/modules/logos_page/GitHub-Mark.png' width=100>

**Everything is available in this repository!**

`git clone https://github.com/Rhoana/perception.git`

`cd perception`

### Setup

We need Anaconda or Miniconda (tested version 5.0.1 on Linux)! Get it [here](https://www.anaconda.com/download/)

The virtual environment with all dependencies (keras, tensorflow, yadayada..) can then be created like this:

`conda env create -f CONDAENV`

The environment can then be directly activated

`conda activate CP`

And now `jupyter notebook` allows for execution of the `IPY/` stuff! Or the driver codes can be run directly from the `EXP/` folder.

### Starting Points

* For the 'elementary perceptual tasks': https://github.com/Rhoana/perception/blob/master/IPY/Figure1.ipynb

* And for 'position-angle': https://github.com/Rhoana/perception/blob/master/IPY/Figure3.ipynb

* And for 'position-length': https://github.com/Rhoana/perception/blob/master/IPY/Figure4.ipynb

* And for the 'bars and framed rectangle' experiment: https://github.com/Rhoana/perception/blob/master/IPY/Figure12.ipynb

* And finally, for 'Webers Law': https://github.com/Rhoana/perception/blob/master/IPY/Weber_Fechner_Law.ipynb

* **Also, here is the code for training and testing the regression:** https://github.com/Rhoana/perception/blob/master/EXP/run_regression.py

* Other good stuff is hidden in IPY/ and EXP/ - please browse :)

## BibTex

```
@article{haehn2018evaluating,
  title={Evaluating 'Graphical Perception' with CNNs},
  author={Haehn, Daniel and Tompkin, James and Pfister, Hanspeter},
  journal={IEEE Transactions on Visualization and Computer Graphics (IEEE VIS)},
  volume={to appear},
  number={X},
  pages={X--X},
  year={2018},
  month={October},
  publisher={IEEE},
  supplemental={http://danielhaehn.com/papers/haehn2018evaluating_supplemental.pdf},
  code={http://rhoana.org/perception/},
  data={http://rhoana.org/perception/},
  video={https://vimeo.com/280506639},
  fastforward={https://vimeo.com/285106317}
}
```

### Feedback..

.. is very welcome! Please contact http://danielhaehn.com !

<a target=_blank href='http://vcg.seas.harvard.edu'><img src='https://pbs.twimg.com/profile_images/851447292120805376/y_RzZDR__400x400.jpg' width=100></a>

**THANK YOU**
