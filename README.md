# Locate Nano Particles


## What is this

`nplocate` is a custom script I wrote to locate very tiny particles from a confocal image. These images often suffered from extreme influence of the [PSF](https://en.wikipedia.org/wiki/Point_spread_function), even after very detailed and completed deconvolution.

To squeeze a bit more information out of these highly distorted data, I wrote this code to effectly "fit" entire 3D image.


## The idea

This is not a fully functional particle tracking package like [trackpy](https://github.com/soft-matter/trackpy) or [colloids](https://github.com/MathieuLeocmach/colloids). Instead, think of `nplocate` as an <big>extension</big> of current tracking packages.

The logic behind the code is quite simple. The arguments are,

1. It is *easy* to find *some* particles, even in a highly distorted image.
1. If we know the locations of some particles ({**r**}), we can measure their average shape (**S**).
2. With {**r**} and **S**, we can simulate a "fake image"
3. We can find previously unfound particles in the difference between the real image and fake image. 


## Installing the code

The simplest way is

```
pip install nplocate
```

You can also download this repository, and use the following command to install the code

```
pip install .
```

## Using the code


There are some notebooks in the folder `example` that introduced how to use this package, along with [trackpy](https://github.com/soft-matter/trackpy). 
