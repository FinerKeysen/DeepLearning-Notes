#!/usr/bin/env python
# coding=utf-8

from PIL import Image
from numpy import *
from pylab import *
import harris 

im = array(Image.open("../../demos/image001.jpg").convert('L'))
harrism = harris.compute_harris_response(im)
filtered_coords = harris.get_harris_points(harrism, 6)
harris.plot_harris_points(im, filtered_coords)
