# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 00:17:02 2017

@author: Ouch
"""

import sys
from PIL import Image

inputFile = str(sys.argv[1])
image1 = Image.open(inputFile)
#image1 = Image.open('westbrook.jpg')

pixel = image1.load()
r, g, b = image1.split()
rpixel = r.load()
gpixel = g.load()
bpixel = b.load()

#Every Column
for i in range(image1.size[0]):
    #Every Row
    for j in range(image1.size[1]):
        r2 = int(rpixel[i,j]//2)
        g2 = int(gpixel[i,j]//2)
        b2 = int(bpixel[i,j]//2)
        pixel[i,j] = (r2, g2, b2)
        image1.putpixel((i,j), pixel[i,j])
image1.save('Q2.png')
