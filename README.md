# Digit Clutter

Contains python scripts for generating digit clutter and digit debris stimuli as described [here](https://doi.org/10.1101/133330)

## Prerequisites 

Requires the following python packages, scipy, numpy and Pillow. These should be installed if your using the standard Anaconda distribution. If not these will need to be installed with the following command

```
pip install scipy numpy Pillow
```

Also requires ImageMagick. The download and installation instructions can be found [here](http://www.imagemagick.org/script/download.php).

## Basic usage

Firstly, just ensure your files are in a directory on your python path. Having a look at the following working [example](example_script/test_clutter_code.ipynb) should give an idea of how to generate stimulus sets.

A [script](example_script/light_debris_generator.py) is also included that generates 1000 images with the same attributes as the light debris image set described in the paper.