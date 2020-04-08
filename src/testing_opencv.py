#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 09:55:45 2020

@author: ithier
"""

import cv2
import sklearn
import numpy as np

def main():
    img = cv2.imread("../images/0.jpg")

    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()