import os as os
import cv2 as cv
import numpy as np
import pandas as pd
import glob
from fpdf import FPDF
import matplotlib.pyplot as plt

imlist = []
pdf = FPDF()
dfy = pd.DataFrame(np.zeros((1, 1)))
dfy = dfy.add(30)
filelist = []
X = np.linspace(1,900,900)
Y = np.linspace(1,900,900)


for file in glob.glob(r"C:\Users\Andrew Hu\Dropbox\PC\Desktop\*.jpg"):
#for file in glob.glob(r"C:\Users\Andrew Hu\Dropbox\PC\Desktop\val-pre-delivery\*.jpg"):
    IM = cv.imread(file)
    imlist.append(IM)
    filelist.append(file)


#print(imlist[1])
for i in range(0, len(imlist)):
    img = imlist[i]
    scale_percent = 15  # enter percentage of original size
    buffer = 4*scale_percent//10  # set to 0 for the standard cropping in opencv
    theta = -0.12  # in degrees, tells python how far to rotate CCW


    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    imgr = cv.resize(img, dim, interpolation=cv.INTER_AREA)

    # alpha from 1.0-3.0 #beta from 0-100
    newimg = cv.convertScaleAbs(imgr, alpha=0.57, beta=38)
    height, width = img.shape[0], img.shape[1]
    print("width: {}, height: {}".format(width, height))

    ## lines 38-42 finds contours in grayscale
    imgray = cv.cvtColor(newimg, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv.findContours(
            thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    ## lines 44-52 iterate through contours, lengths of contour, and index the largest one (actual module)
    size = []
    for j in range(0, len(contours)):
        size.append(len(contours[j]))
    print(size)
    print(max(size))
    print(size.index(max(size)))
    largest = size.index(max(size))
    cnt = contours[largest]  # cnt most important variable, using this a ton





################################################################
    ## below is to crop based on the intensity values at halfway up the height

    if imgr.shape[0] > 500:
        ydist = imgr.shape[1]//2 #if you dont do floor it becomes an int not float
        print(ydist)
        for j in range(0,900):
            Y[j] = imgray[ydist][j]
            #dimensions of imgray are 600 by 900

        #Valarray = np.where(Y>threshold)
        #print(Valarray)
        #Valarray = np.asarray(Valarray)
        #print(Valarray[0][10:20])

        plt.figure()
        plt.title("Grayscale Intensity")
        plt.xlabel("x-position")
        plt.ylabel("Pixel Intensity (grayscale)")
        plt.plot(Y)
        plt.xlim([0, 900])


        threshold = max(Y)*0.72
        dark = np.where(Y>threshold)
        DarkTF = Y>threshold
        count = 0
        badend = 0
        badsta = 0
        for n in range(450,len(DarkTF)):
            #print(np.asarray(DarkTF[n]))
            if np.asarray(DarkTF[n])==True:
                count = count+1
            else:
                badend = badend+1
                if badend >= 10:
                    count = 0
                    foundend = n
                    break
        count = 0
        for m in range(450,1,-1):
            if np.asarray(DarkTF[m])==True:
                count = count+1
            else:
                badsta = badsta+1
                if badsta >= 10:
                    count = 0
                    foundsta = m
                    break
        print(foundend)
        print(foundsta)
        #count is the number of trues in a rows
        #bad is the number of falses in a row, the if statement is the buffer amount
        #for m in range(450,1,-1):

    ################################################################
    #VERY IMPORTANT, THE PART THAT ACTUALLY CROPS
        (x, y, w, h) = cv.boundingRect(cnt)
        #cropped = imgr[y-buffer:y+h+buffer, x-buffer:x+w+buffer]
        cropped = imgr[y-buffer:y+h+buffer, foundsta:foundend]
        # cv2.warpAffine expects shape in (length, height)
        shape = (cropped.shape[1], cropped.shape[0])
        matrix = cv.getRotationMatrix2D(center=(x, y), angle=theta, scale=1)
        cropped = cv.warpAffine(src=cropped, M=matrix, dsize=shape)
        xnew = int(x - width/2)
        ynew = int(y - height/2)
        cropped = cropped[ynew:ynew+height, xnew:xnew+width]
        w = foundend-350
        #print(y, y+h, x, x+w)
    else:
        cropped = imgr

################################################################

    ## below visualize the contours to help with debugging
    #cv.drawContours(newimg, contours, -1, (0, 255, 0), 3)  # draws all contours found
    #cv.drawContours(newimg, [cnt], 0, (0, 255, 0), 3)  # draws the largest contour
    cv.rectangle(imgr,(x,y),(x+w,y+h),(255,0,0),3) #draws the rectangle found
    #cv.imshow('Working Image' + filelist[i][30:], newimg)
    cv.rectangle(imgr,(x,ydist),(x+w,ydist),(255,0,0),3) #draws the rectangle plotting
    #cv.imshow('Output Image', imgr)
    #cv.imshow('CROP', cropped)
    plt.show()
    cv.waitKey(0)

    ##below is lines needed to add the images to a pdf, do pdf.add_page, pdf.image, pdf.output
    pathname = r'C:\Users\Andrew Hu\Dropbox\PC\Downloads\poop' + \
        str(i+2) + '.jpg'  # dummy file name that gets deleted after
    cv.imwrite(pathname, cropped)
    pdf.set_font('Arial', 'B', 12)
    if (i % 2) == 0:
        pdf.add_page()
        pdfy = 20
        pdf.image(pathname, x=0, y=pdfy, w=w/2, h=h/2)
    else:
        pdfy = 95
        pdf.image(pathname, x=0, y=pdfy, w=w/2, h=h/2)
    #pdf.image(pathname, x=0, y=pdfy, w=w/2, h=h/2)
    os.remove(pathname)  # delete file name once finished
    pdf.multi_cell(0, h/2, 'VAL: ' + filelist[i][55:])
pdf.output("yourfile1.pdf", "F")
