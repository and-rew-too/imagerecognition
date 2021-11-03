import os as os
import cv2 as cv
import numpy as np
import pandas as pd
import glob
from fpdf import FPDF

imlist = []
pdf = FPDF()
dfy = pd.DataFrame(np.zeros((1, 1)))
dfy = dfy.add(30)

for img in glob.glob(r"C:\Users\Andrew Hu\Dropbox\PC\Desktop\*.jpg"):
    n = cv.imread(img)
    imlist.append(n)

#print(imlist[1])
for i in range(0, len(imlist)):
    img = imlist[i]
    scale_percent = 15  # enter percentage of original size
    buffer = 7*scale_percent//10  # set to 0 for the standard cropping in opencv
    theta = 0.05  # in degrees, tells python how far to rotate CCW

    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    imgr = cv.resize(img, dim, interpolation=cv.INTER_AREA)

    # alpha from 1.0-3.0 #beta from 0-100
    newimg = cv.convertScaleAbs(imgr, alpha=1.5, beta=70)
    height, width = img.shape[0], img.shape[1]
    print("width: {}, height: {}".format(width, height))

    ## line 34-37 finds contours in grayscale
    imgray = cv.cvtColor(newimg, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv.findContours(
            thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    ## line 41-49 iterate through contours, lengths of contour, then index the largest one (the actual module)
    size = []
    for j in range(0, len(contours)):
        size.append(len(contours[j]))
    print(size)
    print(max(size))
    print(size.index(max(size)))
    largest = size.index(max(size))
    cnt = contours[largest]  # cnt most important variable, using this a ton

    (x, y, w, h) = cv.boundingRect(cnt)
    cropped = imgr[y-buffer:y+h+buffer, x-buffer:x+w+buffer]
    # cv2.warpAffine expects shape in (length, height)
    shape = (cropped.shape[1], cropped.shape[0])
    matrix = cv.getRotationMatrix2D(center=(x, y), angle=theta, scale=1)
    cropped = cv.warpAffine(src=cropped, M=matrix, dsize=shape)
    xnew = int(x - width/2)
    ynew = int(y - height/2)
    cropped = cropped[ynew:ynew+height, xnew:xnew+width]

    ## below visualize the contours to help with debugging
    #cv.drawContours(newimg, contours, -1, (0,255,0), 3) #draws all contours found
    cv.drawContours(newimg, [cnt], 0, (0, 255, 0),
                    3)  # draws the largest contour
    #cv.rectangle(imgr,(x,y),(x+w,y+h),(255,0,0),3) #draws the rectangle found

    #cv.imshow('Working Image', newimg)
    #cv.imshow('Output Image', imgr)
    #cv.imshow('CROP', cropped)
    cv.waitKey(0)

    ##below is lines needed to add the images to a pdf, do pdf.add_page, pdf.image, pdf.output
    pathname = r'C:\Users\Andrew Hu\Dropbox\PC\Downloads\poop' + \
        str(h) + '.jpg'  # dummy file name that gets deleted after
    cv.imwrite(pathname, cropped)

    dfy.loc[i+1] = h
    if (i % 2) == 0:
        pdf.add_page()
        pdfy = 3*dfy.loc[i]
    else:
        pdfy = 0
    pdf.image(pathname, x=0, y=pdfy, w=w/3, h=h/3)
    os.remove(pathname)  # delete file name once finished
    pdf.set_font('Arial', 'B', 16)
    pdf.multi_cell(0, (h)+20, 'test asdfaf')
pdf.output("yourfile.pdf", "F")
