import os as os
import cv2 as cv
import numpy as np
import pandas as pd
import glob
from fpdf import FPDF


imlist = []
pdf = FPDF()
filelist = []

for file in glob.glob(r"C:\Users\Andrew Hu\Dropbox\PC\Desktop\val-post-delivery\*.jpg"):
    IM = cv.imread(file)
    imlist.append(IM)
    filelist.append(file)

for i in range(0, len(imlist)):
    img = imlist[i]

    scale_percent = 15
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    imgr = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    imgr = cv.flip(src=imgr, flipCode=1)

    ##below is lines needed to add the images to a pdf, do pdf.add_page, pdf.image, pdf.output
    pathname = r'C:\Users\Andrew Hu\Dropbox\PC\Downloads\pee' + \
        str(i+2) + '.jpg'  # dummy file name that gets deleted after
    cv.imwrite(pathname, imgr)
    # below is to add single photo per page (above is two photos per page)
    pdf.add_page()
    pdf.image(pathname, x=0, y=20, w=width/6, h=height/6)
    pdf.set_font('Arial', 'B', 12)
    pdf.multi_cell(0, height/7, 'VAL: ' + filelist[i][56:67] + '-A&B')
    os.remove(pathname)  # delete file name once finished

pdf.output("yourfile2.pdf", "F")
