import numpy as np
import skimage as sk
import skimage.io as skio
import skimage.color as color
import os
import matplotlib.pyplot as plt
import single_scale as ss
import multi_scale as ms
import argparse

parser = argparse.ArgumentParser(description='main function for Project 1')
parser.add_argument('--single_scale', '-s',
                    dest="single_scale",
                    help =  'whether to run the single scale method',
                    default = "True")
parser.add_argument('--multi_scale', '-m',
                    dest="multi_scale",
                    help =  'whether to run the image pyramid method',
                    default = "True")
parser.add_argument('--extra', '-e',
                    dest="extra",
                    help =  'whether to run image pyramid method on extra images',
                    default = "True")

args = parser.parse_args()

if eval(args.single_scale):
    print("single scale")
    ss.main()
if eval(args.multi_scale):
    print("multi scale")
    ms.main()
if eval(args.extra):
    print("extra")
    ms.extra()
