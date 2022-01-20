"""
===========================================
CMOS camera gain analysis program ver.1.0
===========================================
*By: so-nano-car inspired by Mr. "Apranat"*
*Reference: https://www.mirametrics.com/tech_note_ccdgain.php *
*License: BSD*

*Revision history*
ver 1.0 New release
"""

import collections
import matplotlib.pyplot as plt
import numpy as np
import glob
from astropy.visualization import astropy_mpl_style
from astropy.io import fits
import time
import sys
from numpy.lib import savetxt
from sklearn.metrics import r2_score
import gc

# Load sensor pixel size (array size)
def getSenserPixelSize(fitspath):
    imdata = fits.getdata(glob.glob(fitspath)[0], ext = 0) # Get the x,y size of the first fits file found.
    array_y = np.shape(imdata)[0]
    array_x = np.shape(imdata)[1]
    return(array_x, array_y)

# Load files and number of files: n, then store to 3-D numpy array
def loadFilesStore3DnumpyArray(fitspath, array_x, array_y):
    fitsFiles = glob.glob (fitspath)
    fitsNum   = len(fitsFiles)
    stack = np.empty((fitsNum, array_y, array_x), dtype = np.int32)
    gainstack = np.empty((fitsNum,1), dtype = np.int16)
    for i, img in enumerate(fitsFiles):
            imdata = fits.getdata(img, ext = 0)
            stack[i,:,:] = imdata[:,:]
            hdul = fits.open(img)
            gain = hdul[0].header['GAIN']
            gainstack[i] = gain
            print(f"loadFitsFiles {i+1}/{fitsNum}")
    gainlist = gainstack.flatten().tolist()
    count = collections.Counter(gainlist)
    gainkey = np.array([*count.keys()]) #list of gain values
    gainvalue = np.array([*count.values()]) #number of files with same gain

    stack1_temp = np.zeros((fitsNum,array_y,array_x), dtype = np.int32)
    for i, img in enumerate(fitsFiles):
        imdata = fits.getdata(img, ext = 0)
        hdul = fits.open(img)
        gain = hdul[0].header['GAIN']
        try:
            if gain == gainkey[0]:
                stack1_temp[i,:,:] = imdata[:,:]
        except IndexError:
            pass
    if np.all(stack1_temp == 0) == False:
        stack1 =  stack1_temp[0:gainvalue[0],:,:]
    if np.all(stack1_temp == 0) == True:
        stack1 = stack1_temp
    del stack1_temp
    gc.collect()
       
    stack2_temp = np.zeros((fitsNum,array_y,array_x), dtype = np.int32)
    for i, img in enumerate(fitsFiles):
        imdata = fits.getdata(img, ext = 0)
        hdul = fits.open(img)
        gain = hdul[0].header['GAIN']
        try:
            if gain == gainkey[1]:
                stack2_temp[i,:,:] = imdata[:,:]
        except IndexError:
            pass
    if np.all(stack2_temp == 0) == False:
        stack2 = stack2_temp[gainvalue[0] : gainvalue[0] + gainvalue[1],:,:]
    if np.all(stack2_temp == 0) == True:
        stack2 = stack2_temp
    del stack2_temp
    gc.collect()
    
    stack3_temp = np.zeros((fitsNum,array_y,array_x), dtype = np.int32)
    for i, img in enumerate(fitsFiles):
        imdata = fits.getdata(img, ext = 0)
        hdul = fits.open(img)
        gain = hdul[0].header['GAIN']
        try:
            if gain == gainkey[2]:
                stack3_temp[i,:,:] = imdata[:,:]
        except IndexError:
            pass
    if np.all(stack3_temp == 0) == False:
        stack3 = stack3_temp[gainvalue[0] + gainvalue[1] : gainvalue[0] + gainvalue[1] + gainvalue[2],:,:]
    if np.all(stack3_temp == 0) == True:
        stack3 = stack3_temp
    del stack3_temp
    gc.collect()
    
    stack4_temp = np.zeros((fitsNum,array_y,array_x), dtype = np.int32)
    for i, img in enumerate(fitsFiles):
        imdata = fits.getdata(img, ext = 0)
        hdul = fits.open(img)
        gain = hdul[0].header['GAIN']
        try:
            if gain == gainkey[3]:
                stack4_temp[i,:,:] = imdata[:,:]
        except IndexError:
            pass
    if np.all(stack4_temp == 0) == False:
        stack4 = stack4_temp[gainvalue[0] + gainvalue[1] + gainvalue[2]: gainvalue[0] + gainvalue[1] + gainvalue[2] + gainvalue[3],:,:]
    if np.all(stack4_temp == 0) == True:
        stack4 = stack4_temp
    del stack4_temp
    gc.collect()

    stack5_temp = np.zeros((fitsNum,array_y,array_x), dtype = np.int32)
    for i, img in enumerate(fitsFiles):
        imdata = fits.getdata(img, ext = 0)
        hdul = fits.open(img)
        gain = hdul[0].header['GAIN']
        try:
            if gain == gainkey[4]:
                stack5_temp[i,:,:] = imdata[:,:]
        except IndexError:
            pass
    if np.all(stack5_temp == 0) == False:
        stack5 = stack5_temp[gainvalue[0] + gainvalue[1] + gainvalue[2] + gainvalue[3]: gainvalue[0] + gainvalue[1] + gainvalue[2] + gainvalue[3] + gainvalue[4],:,:]
    if np.all(stack5_temp == 0) == True:
        stack5 = stack5_temp
    del stack5_temp
    gc.collect()

    stack6_temp = np.zeros((fitsNum,array_y,array_x), dtype = np.int32)
    for i, img in enumerate(fitsFiles):
        imdata = fits.getdata(img, ext = 0)
        hdul = fits.open(img)
        gain = hdul[0].header['GAIN']
        try:
            if gain == gainkey[5]:
                stack6_temp[i,:,:] = imdata[:,:]
        except IndexError:
            pass
    if np.all(stack6_temp == 0) == False:
        stack6 = stack6_temp[gainvalue[0] + gainvalue[1] + gainvalue[2] + gainvalue[3] + gainvalue[4]: gainvalue[0] + gainvalue[1] + gainvalue[2] + gainvalue[3] + gainvalue[4] + gainvalue[5],:,:]
    if np.all(stack6_temp == 0) == True:
        stack6 = stack6_temp
    del stack6_temp
    gc.collect()

    stack7_temp = np.zeros((fitsNum,array_y,array_x), dtype = np.int32)
    for i, img in enumerate(fitsFiles):
        imdata = fits.getdata(img, ext = 0)
        hdul = fits.open(img)
        gain = hdul[0].header['GAIN']
        try:
            if gain == gainkey[6]:
                stack7_temp[i,:,:] = imdata[:,:]
        except IndexError:
            pass
    if np.all(stack7_temp == 0) == False:
        stack7 = stack7_temp[gainvalue[0] + gainvalue[1] + gainvalue[2] + gainvalue[3] + gainvalue[4] + gainvalue[5]: gainvalue[0] + gainvalue[1] + gainvalue[2] + gainvalue[3] + gainvalue[4] + gainvalue[5] + gainvalue[6],:,:]
    if np.all(stack7_temp == 0) == True:
        stack7 = stack7_temp
    del stack7_temp
    gc.collect()

    stack8_temp = np.zeros((fitsNum,array_y,array_x), dtype = np.int32)
    for i, img in enumerate(fitsFiles):
        imdata = fits.getdata(img, ext = 0)
        hdul = fits.open(img)
        gain = hdul[0].header['GAIN']
        try:
            if gain == gainkey[7]:
                stack8_temp[i,:,:] = imdata[:,:]
        except IndexError:
            pass
    if np.all(stack8_temp == 0) == False:
        stack8 = stack8_temp[gainvalue[0] + gainvalue[1] + gainvalue[2] + gainvalue[3] + gainvalue[4] + gainvalue[5] + gainvalue[6]: gainvalue[0] + gainvalue[1] + gainvalue[2] + gainvalue[3] + gainvalue[4] + gainvalue[5] + gainvalue[6] + gainvalue[7],:,:]
    if np.all(stack8_temp == 0) == True:
        stack8 = stack8_temp
    del stack8_temp
    gc.collect()

    stack9_temp = np.zeros((fitsNum,array_y,array_x), dtype = np.int32)
    for i, img in enumerate(fitsFiles):
        imdata = fits.getdata(img, ext = 0)
        hdul = fits.open(img)
        gain = hdul[0].header['GAIN']
        try:
            if gain == gainkey[8]:
                stack9_temp[i,:,:] = imdata[:,:]
        except IndexError:
            pass
    if np.all(stack9_temp == 0) == False:
        stack9 = stack9_temp[gainvalue[0] + gainvalue[1] + gainvalue[2] + gainvalue[3] + gainvalue[4] + gainvalue[5] + gainvalue[6] + gainvalue[7]: gainvalue[0] + gainvalue[1] + gainvalue[2] + gainvalue[3] + gainvalue[4] + gainvalue[5] + gainvalue[6] + gainvalue[7] + gainvalue[8],:,:]
    if np.all(stack9_temp == 0) == True:
        stack9 = stack9_temp
    del stack9_temp
    gc.collect()

    return (stack, stack1, stack2, stack3, stack4, stack5, stack6, stack7, stack8, stack9, gainstack, fitsNum, gainkey, gainvalue)

# Calculate average lightness and variance for each flat frame: method 1
def calculate_flat_1(stack1, stack2, stack3, stack4, stack5, stack6, stack7, stack8, stack9, fitsNum, gainvalue):

    stack1_14 = np.float32(stack1 / 4)
    ave1 = np.mean(np.mean(stack1_14, axis = 1), axis = 1)
    if np.all(ave1 == 0) == False:
         flat_ave1 = ave1[np.nonzero(ave1)]       
    if np.all(ave1 == 0) == True:
        flat_ave1 = ave1
        flat_var1 = ave1
    var1 = np.zeros((fitsNum))
    try:
        for i in range (gainvalue[0]):
            var1[i] = np.var(stack1_14[i,:,:])
        flat_var1 = var1[np.nonzero(var1)]
    except IndexError:
        pass
    if np.all(flat_var1 == 0) == False:
        fit1 = np.polyfit(flat_ave1, flat_var1, 1)
        val1 = np.polyval(fit1,flat_ave1)
        r2_1 = r2_score(flat_var1, val1)
    print(f'fit1 = {fit1}')
    if np.all(flat_var1 == 0) == True:
        fit1 = np.zeros((2,1))
        r2_1 = 0
    
    stack2_14 = np.float32(stack2 / 4)
    ave2 = np.mean(np.mean(stack2_14, axis = 1), axis = 1)
    if np.all(ave2 == 0) == False:
        flat_ave2 = ave2[np.nonzero(ave2)]
    if np.all(ave2 == 0) == True:
        flat_ave2 = ave2
        flat_var2 = ave2
    var2 = np.zeros((fitsNum))
    try:
        for i in range (gainvalue[1]):
            var2[i] = np.var(stack2_14[i,:,:])
        flat_var2 = var2[np.nonzero(var2)]
    except IndexError:
        pass
    if np.all(flat_var2 == 0) == False:
        fit2 = np.polyfit(flat_ave2, flat_var2, 1)
        val2 = np.polyval(fit2,flat_ave2)
        r2_2 = r2_score(flat_var2, val2)
    if np.all(flat_var2 == 0) == True:
        fit2 = np.zeros((2,1))
        r2_2 = 0
    
    stack3_14 = np.float32(stack3 / 4)
    ave3 = np.mean(np.mean(stack3_14, axis = 1), axis = 1)
    if np.all(ave3 == 0) == False:
        flat_ave3 = ave3[np.nonzero(ave3)]
    if np.all(ave3 == 0) == True:
        flat_ave3 = ave3
        flat_var3 = ave3
    var3 = np.zeros((fitsNum))
    try:
        for i in range (gainvalue[2]):
            var3[i] = np.var(stack3_14[i,:,:])
        flat_var3 = var3[np.nonzero(var3)]
    except IndexError:
        pass
    if np.all(flat_var3 == 0) == False:
        fit3 = np.polyfit(flat_ave3, flat_var3, 1)
        val3 = np.polyval(fit3,flat_ave3)
        r2_3 = r2_score(flat_var3, val3)
    if np.all(flat_var3 == 0) == True:
        fit3 = np.zeros((2,1))
        r2_3 = 0

    stack4_14 = np.float32(stack4 / 4)
    ave4 = np.mean(np.mean(stack4_14, axis = 1), axis = 1)
    if np.all(ave4 == 0) == False:
        flat_ave4 = ave4[np.nonzero(ave4)]
    if np.all(ave4 == 0) == True:
        flat_ave4 = ave4
        flat_var4 = ave4
    var4 = np.zeros((fitsNum))
    try:
        for i in range (gainvalue[3]):
            var4[i] = np.var(stack4_14[i,:,:])
        flat_var4 = var4[np.nonzero(var4)]
    except IndexError:
        pass
    if np.all(flat_var4 == 0) == False:
        fit4 = np.polyfit(flat_ave4, flat_var4, 1)
        val4 = np.polyval(fit4,flat_ave4)
        r2_4 = r2_score(flat_var4, val4)
    if np.all(flat_var4 == 0) == True:
        fit4 = np.zeros((2,1))
        r2_4 = 0

    stack5_14 = np.float32(stack5 / 4)
    ave5 = np.mean(np.mean(stack5_14, axis = 1), axis = 1)
    if np.all(ave5 == 0) == False:
        flat_ave5 = ave5[np.nonzero(ave5)]
    if np.all(ave5 == 0) == True:
        flat_ave5 = ave5
        flat_var5 = ave5
    var5 = np.zeros((fitsNum))
    try:
        for i in range (gainvalue[4]):
            var5[i] = np.var(stack5_14[i,:,:])
        flat_var5 = var5[np.nonzero(var5)]
    except IndexError:
        pass
    if np.all(flat_var5 == 0) == False:
        fit5 = np.polyfit(flat_ave5, flat_var5, 1)
        val5 = np.polyval(fit5,flat_ave5)
        r2_5 = r2_score(flat_var5, val5)
    if np.all(flat_var5 == 0) == True:
        fit5 = np.zeros((2,1))
        r2_5 = 0

    stack6_14 = np.float32(stack6 / 4)
    ave6 = np.mean(np.mean(stack6_14, axis = 1), axis = 1)
    if np.all(ave6 == 0) == False:
        flat_ave6 = ave6[np.nonzero(ave6)]
    if np.all(ave6 == 0) == True:
        flat_ave6 = ave6
        flat_var6 = ave6
    var6 = np.zeros((fitsNum))
    try:
        for i in range (gainvalue[5]):
            var6[i] = np.var(stack6_14[i,:,:])
        flat_var6 = var6[np.nonzero(var6)]
    except IndexError:
        pass
    if np.all(flat_var6 == 0) == False:
        fit6 = np.polyfit(flat_ave6, flat_var6, 1)
        val6 = np.polyval(fit6,flat_ave6)
        r2_6 = r2_score(flat_var6, val6)
    if np.all(flat_var6 == 0) == True:
        fit6 = np.zeros((2,1))
        r2_6 = 0

    stack7_14 = np.float32(stack7 / 4)
    ave7 = np.mean(np.mean(stack7_14, axis = 1), axis = 1)
    if np.all(ave7 == 0) == False:
        flat_ave7 = ave7[np.nonzero(ave7)]
    if np.all(ave7 == 0) == True:
        flat_ave7 = ave7
        flat_var7 = ave7
    var7 = np.zeros((fitsNum))
    try:
        for i in range (gainvalue[6]):
            var7[i] = np.var(stack7_14[i,:,:])
        flat_var7 = var7[np.nonzero(var7)]
    except IndexError:
        pass
    if np.all(flat_var7 == 0) == False:
        fit7 = np.polyfit(flat_ave7, flat_var7, 1)
        val7 = np.polyval(fit7,flat_ave7)
        r2_7 = r2_score(flat_var7, val7)
    if np.all(flat_var7 == 0) == True:
        fit7 = np.zeros((2,1))
        r2_7 = 0

    stack8_14 = np.float32(stack8 / 4)
    ave8 = np.mean(np.mean(stack8_14, axis = 1), axis = 1)
    if np.all(ave8 == 0) == False:
        flat_ave8 = ave8[np.nonzero(ave8)]
    if np.all(ave8 == 0) == True:
        flat_ave8 = ave8
        flat_var8 = ave8
    var8 = np.zeros((fitsNum))
    try:
        for i in range (gainvalue[7]):
            var8[i] = np.var(stack8_14[i,:,:])
        flat_var8 = var8[np.nonzero(var8)]
    except IndexError:
        pass    
    if np.all(flat_var8 == 0) == False:
        fit8 = np.polyfit(flat_ave8, flat_var8, 1)
        val8 = np.polyval(fit8,flat_ave8)
        r2_8 = r2_score(flat_var8, val8)
    if np.all(flat_var8 == 0) == True:
        fit8 = np.zeros((2,1))
        r2_8 = 0

    stack9_14 = np.float32(stack9 / 4)
    ave9 = np.mean(np.mean(stack9_14, axis = 1), axis = 1)
    if np.all(ave9 == 0) == False:
        flat_ave9 = ave9[np.nonzero(ave9)]
    if np.all(ave9 == 0) == True:
        flat_ave9 = ave9
        flat_var9 = ave9
    var9 = np.zeros((fitsNum))
    try:
        for i in range (gainvalue[8]):
            var9[i] = np.var(stack9_14[i,:,:])
        flat_var9 = var9[np.nonzero(var9)]
    except IndexError:
        pass    
    if np.all(flat_var9 == 0) == False:
        fit9 = np.polyfit(flat_ave9, flat_var9, 1)
        val9 = np.polyval(fit9,flat_ave9)
        r2_9 = r2_score(flat_var9, val9)
    if np.all(flat_var9 == 0) == True:
        fit9 = np.zeros((2,1))
        r2_9 = 0

    return (flat_ave1, flat_var1, fit1, r2_1, flat_ave2, flat_var2, fit2, r2_2, flat_ave3, flat_var3, fit3, r2_3, flat_ave4, flat_var4, fit4, r2_4, flat_ave5, flat_var5, fit5, r2_5, flat_ave6, flat_var6, fit6, r2_6, flat_ave7, flat_var7, fit7, r2_7, flat_ave8, flat_var8, fit8, r2_8, flat_ave9, flat_var9, fit9, r2_9)

# Calculate average lightness and variance for each flat frame: method 2
def calculate_flat_2(stack1, stack2, stack3, stack4, stack5, stack6, stack7, stack8, stack9, fitsNum, gainvalue):

    stack1_14 = np.float32(stack1 / 4)
    ave1 = np.mean(np.mean(stack1_14, axis = 1), axis = 1)
    if np.all(ave1 == 0) == False:
        flat_ave1 = np.zeros((fitsNum,1))
        flat_var1 = np.zeros((fitsNum,1))
        for j in range (0, gainvalue[0]-1, 2):
            aveA1 = np.mean(np.mean(stack1_14[j,:,:], axis = 1), axis = 0)
            aveB1 = np.mean(np.mean(stack1_14[j + 1,:,:], axis = 1), axis = 0)
            r1 = aveA1 / aveB1
            imgB1 = r1 * stack1_14[j+1,:,:]
            imgA1 = stack1_14[j,:,:] - imgB1
            flat_ave1[j] = aveA1
            flat_var1[j] = np.var(imgA1) / 2
        flat_ave1 = flat_ave1[np.nonzero(flat_ave1)]
        flat_var1 = flat_var1[np.nonzero(flat_var1)]
    if np.all(ave1 == 0) == True:
        flat_ave1 = ave1
        flat_var1 = ave1
    if np.all(flat_var1 == 0) == False:
        fit1 = np.polyfit(flat_ave1, flat_var1, 1)
        val1 = np.polyval(fit1,flat_ave1)
        r2_1 = r2_score(flat_var1, val1)
    if np.all(flat_var1 == 0) == True:
        fit1 = np.zeros((2,1))
        r2_1 = 0
    
    stack2_14 = np.float32(stack2 / 4)
    ave2 = np.mean(np.mean(stack2_14, axis = 1), axis = 1)
    if np.all(ave2 == 0) == False:
        flat_ave2 = np.zeros((fitsNum,1))
        flat_var2 = np.zeros((fitsNum,1))
        for j in range (0, gainvalue[1]-1, 2):
            aveA2 = np.mean(np.mean(stack2_14[j,:,:], axis = 1), axis = 0)
            aveB2 = np.mean(np.mean(stack2_14[j + 1,:,:], axis = 1), axis = 0)
            r2 = aveA2 / aveB2
            imgB2 = r2 * stack2_14[j+1,:,:]
            imgA2 = stack2_14[j,:,:] - imgB2
            flat_ave2[j] = aveA2
            flat_var2[j] = np.var(imgA2) / 2
        flat_ave2 = flat_ave2[np.nonzero(flat_ave2)]
        flat_var2 = flat_var2[np.nonzero(flat_var2)]
    if np.all(ave2 == 0) == True:
        flat_ave2 = ave2
        flat_var2 = ave2
    if np.all(flat_var2 == 0) == False:
        fit2 = np.polyfit(flat_ave2, flat_var2, 1)
        val2 = np.polyval(fit2,flat_ave2)
        r2_2 = r2_score(flat_var2, val2)
    if np.all(flat_var2 == 0) == True:
        fit2 = np.zeros((2,1))
        r2_2 = 0
    
    stack3_14 = np.float32(stack3 / 4)
    ave3 = np.mean(np.mean(stack3_14, axis = 1), axis = 1)
    if np.all(ave3 == 0) == False:
        flat_ave3 = np.zeros((fitsNum,1))
        flat_var3 = np.zeros((fitsNum,1))
        for j in range (0, gainvalue[2] - 1, 2):
            aveA3 = np.mean(np.mean(stack3_14[j,:,:], axis = 1), axis = 0)
            aveB3 = np.mean(np.mean(stack3_14[j + 1,:,:], axis = 1), axis = 0)
            r3 = aveA3 / aveB3
            imgB3 = r3 * stack3_14[j+1,:,:]
            imgA3 = stack3_14[j,:,:] - imgB3
            flat_ave3[j] = aveA3
            flat_var3[j] = np.var(imgA3) / 2
        flat_ave3 = flat_ave3[np.nonzero(flat_ave3)]
        flat_var3 = flat_var3[np.nonzero(flat_var3)]
    if np.all(ave3 == 0) == True:
        flat_ave3 = ave3
        flat_var3 = ave3
    if np.all(flat_var3 == 0) == False:
        fit3 = np.polyfit(flat_ave3, flat_var3, 1)
        val3 = np.polyval(fit3,flat_ave3)
        r2_3 = r2_score(flat_var3, val3)
    if np.all(flat_var3 == 0) == True:
        fit3 = np.zeros((2,1))
        r2_3 = 0

    stack4_14 = np.float32(stack4 / 4)
    ave4 = np.mean(np.mean(stack4_14, axis = 1), axis = 1)
    if np.all(ave4 == 0) == False:
        flat_ave4 = np.zeros((fitsNum,1))
        flat_var4 = np.zeros((fitsNum,1))
        for j in range (0, gainvalue[3] - 1, 2):
            aveA4 = np.mean(np.mean(stack4_14[j,:,:], axis = 1), axis = 0)
            aveB4 = np.mean(np.mean(stack4_14[j + 1,:,:], axis = 1), axis = 0)
            r4 = aveA4 / aveB4
            imgB4 = r4 * stack4_14[j+1,:,:]
            imgA4 = stack4_14[j,:,:] - imgB4
            flat_ave4[j] = aveA4
            flat_var4[j] = np.var(imgA4) / 2
        flat_ave4 = flat_ave4[np.nonzero(flat_ave4)]
        flat_var4 = flat_var4[np.nonzero(flat_var4)]
    if np.all(ave4 == 0) == True:
        flat_ave4 = ave4
        flat_var4 = ave4
    if np.all(flat_var4 == 0) == False:
        fit4 = np.polyfit(flat_ave4, flat_var4, 1)
        val4 = np.polyval(fit4,flat_ave4)
        r2_4 = r2_score(flat_var4, val4)
    if np.all(flat_var4 == 0) == True:
        fit4 = np.zeros((2,1))
        r2_4 = 0

    stack5_14 = np.float32(stack5 / 4)
    ave5 = np.mean(np.mean(stack5_14, axis = 1), axis = 1)
    if np.all(ave5 == 0) == False:
        flat_ave5 = np.zeros((fitsNum,1))
        flat_var5 = np.zeros((fitsNum,1))
        for j in range (0, gainvalue[4] - 1, 2):
            aveA5 = np.mean(np.mean(stack5_14[j,:,:], axis = 1), axis = 0)
            aveB5 = np.mean(np.mean(stack5_14[j + 1,:,:], axis = 1), axis = 0)
            r5 = aveA5 / aveB5
            imgB5 = r5 * stack5_14[j+1,:,:]
            imgA5 = stack5_14[j,:,:] - imgB5
            flat_ave5[j] = aveA5
            flat_var5[j] = np.var(imgA5) / 2
        flat_ave5 = flat_ave5[np.nonzero(flat_ave5)]
        flat_var5 = flat_var5[np.nonzero(flat_var5)]
    if np.all(ave5 == 0) == True:
        flat_ave5 = ave5
        flat_var5 = ave5
    if np.all(flat_var5 == 0) == False:
        fit5 = np.polyfit(flat_ave5, flat_var5, 1)
        val5 = np.polyval(fit5,flat_ave5)
        r2_5 = r2_score(flat_var5, val5)
    if np.all(flat_var5 == 0) == True:
        fit5 = np.zeros((2,1))
        r2_5 = 0

    stack6_14 = np.float32(stack6 / 4)
    ave6 = np.mean(np.mean(stack6_14, axis = 1), axis = 1)
    if np.all(ave6 == 0) == False:
        flat_ave6 = np.zeros((fitsNum,1))
        flat_var6 = np.zeros((fitsNum,1))
        for j in range (0, gainvalue[5] - 1, 2):
            aveA6 = np.mean(np.mean(stack6_14[j,:,:], axis = 1), axis = 0)
            aveB6 = np.mean(np.mean(stack6_14[j + 1,:,:], axis = 1), axis = 0)
            r6 = aveA6 / aveB6
            imgB6 = r6 * stack6_14[j+1,:,:]
            imgA6 = stack6_14[j,:,:] - imgB6
            flat_ave6[j] = aveA6
            flat_var6[j] = np.var(imgA6) / 2
        flat_ave6 = flat_ave6[np.nonzero(flat_ave6)]
        flat_var6 = flat_var6[np.nonzero(flat_var6)]
    if np.all(ave6 == 0) == True:
        flat_ave6 = ave6
        flat_var6 = ave6
    if np.all(flat_var6 == 0) == False:
        fit6 = np.polyfit(flat_ave6, flat_var6, 1)
        val6 = np.polyval(fit6,flat_ave6)
        r2_6 = r2_score(flat_var6, val6)
    if np.all(flat_var6 == 0) == True:
        fit6 = np.zeros((2,1))
        r2_6 = 0

    stack7_14 = np.float32(stack7 / 4)
    ave7 = np.mean(np.mean(stack7_14, axis = 1), axis = 1)
    if np.all(ave7 == 0) == False:
        flat_ave7 = np.zeros((fitsNum,1))
        flat_var7 = np.zeros((fitsNum,1))
        for j in range (0, gainvalue[6] - 1, 2):
            aveA7 = np.mean(np.mean(stack7_14[j,:,:], axis = 1), axis = 0)
            aveB7 = np.mean(np.mean(stack7_14[j + 1,:,:], axis = 1), axis = 0)
            r7 = aveA7 / aveB7
            imgB7 = r7 * stack7_14[j+1,:,:]
            imgA7 = stack7_14[j,:,:] - imgB7
            flat_ave7[j] = aveA7
            flat_var7[j] = np.var(imgA7) / 2
        flat_ave7 = flat_ave7[np.nonzero(flat_ave7)]
        flat_var7 = flat_var7[np.nonzero(flat_var7)]
    if np.all(ave7 == 0) == True:
        flat_ave7 = ave7
        flat_var7 = ave7
    if np.all(flat_var7 == 0) == False:
        fit7 = np.polyfit(flat_ave7, flat_var7, 1)
        val7 = np.polyval(fit7,flat_ave7)
        r2_7 = r2_score(flat_var7, val7)
    if np.all(flat_var7 == 0) == True:
        fit7 = np.zeros((2,1))
        r2_7 = 0

    stack8_14 = np.float32(stack8 / 4)
    ave8 = np.mean(np.mean(stack8_14, axis = 1), axis = 1)
    if np.all(ave8 == 0) == False:
        flat_ave8 = np.zeros((fitsNum,1))
        flat_var8 = np.zeros((fitsNum,1))
        for j in range (0, gainvalue[7] - 1, 2):
            aveA8 = np.mean(np.mean(stack8_14[j,:,:], axis = 1), axis = 0)
            aveB8 = np.mean(np.mean(stack8_14[j + 1,:,:], axis = 1), axis = 0)
            r8 = aveA8 / aveB8
            imgB8 = r8 * stack8_14[j+1,:,:]
            imgA8 = stack8_14[j,:,:] - imgB8
            flat_ave8[j] = aveA8
            flat_var8[j] = np.var(imgA8) / 2
        flat_ave8 = flat_ave8[np.nonzero(flat_ave8)]
        flat_var8 = flat_var8[np.nonzero(flat_var8)]
    if np.all(ave8 == 0) == True:
        flat_ave8 = ave8
        flat_var8 = ave8
    if np.all(flat_var8 == 0) == False:
        fit8 = np.polyfit(flat_ave8, flat_var8, 1)
        val8 = np.polyval(fit8,flat_ave8)
        r2_8 = r2_score(flat_var8, val8)
    if np.all(flat_var8 == 0) == True:
        fit8 = np.zeros((2,1))
        r2_8 = 0

    stack9_14 = np.float32(stack9 / 4)
    ave9 = np.mean(np.mean(stack9_14, axis = 1), axis = 1)
    if np.all(ave9 == 0) == False:
        flat_ave9 = np.zeros((fitsNum,1))
        flat_var9 = np.zeros((fitsNum,1))
        for j in range (0, gainvalue[8] - 1, 2):
            aveA9 = np.mean(np.mean(stack9_14[j,:,:], axis = 1), axis = 0)
            aveB9 = np.mean(np.mean(stack9_14[j + 1,:,:], axis = 1), axis = 0)
            r9 = aveA9 / aveB9
            imgB9 = r9 * stack9_14[j+1,:,:]
            imgA9 = stack9_14[j,:,:] - imgB9
            flat_ave9[j] = aveA9
            flat_var9[j] = np.var(imgA9) / 2
        flat_ave9 = flat_ave9[np.nonzero(flat_ave9)]
        flat_var9 = flat_var9[np.nonzero(flat_var9)]
    if np.all(ave9 == 0) == True:
        flat_ave9 = ave9
        flat_var9 = ave9
    if np.all(flat_var9 == 0) == False:
        fit9 = np.polyfit(flat_ave9, flat_var9, 1)
        val9 = np.polyval(fit9,flat_ave9)
        r2_9 = r2_score(flat_var9, val9)
    if np.all(flat_var9 == 0) == True:
        fit9 = np.zeros((2,1))
        r2_9 = 0

    return (flat_ave1, flat_var1, fit1, r2_1, flat_ave2, flat_var2, fit2, r2_2, flat_ave3, flat_var3, fit3, r2_3, flat_ave4, flat_var4, fit4, r2_4, flat_ave5, flat_var5, fit5, r2_5, flat_ave6, flat_var6, fit6, r2_6, flat_ave7, flat_var7, fit7, r2_7, flat_ave8, flat_var8, fit8, r2_8, flat_ave9, flat_var9, fit9, r2_9)

def plot_flat(flat_ave1, flat_var1, fit1, r2_1, flat_ave2, flat_var2, fit2, r2_2, flat_ave3, flat_var3, fit3, r2_3, flat_ave4, flat_var4, fit4, r2_4, flat_ave5, flat_var5, fit5, r2_5, flat_ave6, flat_var6, fit6, r2_6, flat_ave7, flat_var7, fit7, r2_7, flat_ave8, flat_var8, fit8, r2_8, flat_ave9, flat_var9, fit9, r2_9, gainkey):
    axs = plt.figure(figsize=(10,10))
    ax1 = axs.add_subplot(3,3,1)
    ax1.scatter(flat_ave1, flat_var1, s = 10, marker = '.', color = "red")
    ax1.plot(flat_ave1, flat_ave1 * fit1[0] + fit1[1], linestyle = "dotted")
    ax1.grid(which = "both", linewidth = 0.5, alpha = 0.1)
    try:
        ax1.set_title(f"gain = {gainkey[0]}, slope = {fit1[0]}, r2 = {r2_1}", fontsize = 5)
    except IndexError:
        pass
    ax1.set_xlabel('Luminance', fontsize = 5)
    ax1.set_ylabel('Variance', fontsize = 5)
    ax1.tick_params(labelsize = 7)

    ax2 = axs.add_subplot(3,3,2)
    ax2.scatter(flat_ave2, flat_var2, s = 10, marker = '.', color = "red")
    ax2.plot(flat_ave2, flat_ave2 * fit2[0] + fit2[1], linestyle = "dotted")
    ax2.grid(which = "both", linewidth = 0.5, alpha = 0.1)
    try:
        ax2.set_title(f"gain = {gainkey[1]}, slope = {fit2[0]}, r2 = {r2_2}", fontsize = 5)
    except IndexError:
        pass
    ax2.set_xlabel('Luminance', fontsize = 5)
    ax2.set_ylabel('Variance', fontsize = 5)
    ax2.tick_params(labelsize = 7)

    ax3 = axs.add_subplot(3,3,3)
    ax3.scatter(flat_ave3, flat_var3, s = 10, marker = '.', color = "red")
    ax3.plot(flat_ave3, flat_ave3 * fit3[0] + fit3[1], linestyle = "dotted")
    ax3.grid(which = "both", linewidth = 0.5, alpha = 0.1)
    try:
        ax3.set_title(f"gain = {gainkey[2]}, slope = {fit3[0]}, r2 = {r2_3}", fontsize = 5)
    except IndexError:
        pass
    ax3.set_xlabel('Luminance', fontsize = 5)
    ax3.set_ylabel('Variance', fontsize = 5)
    ax3.tick_params(labelsize = 7)

    ax4 = axs.add_subplot(3,3,4)
    ax4.scatter(flat_ave4, flat_var4, s = 10, marker = '.', color = "red")
    ax4.plot(flat_ave4, flat_ave4 * fit4[0] + fit4[1], linestyle = "dotted")
    ax4.grid(which = "both", linewidth = 0.5, alpha = 0.1)
    try:
        ax4.set_title(f"gain = {gainkey[3]}, slope = {fit4[0]}, r2 = {r2_4}", fontsize = 5)
    except IndexError:
        pass
    ax4.set_xlabel('Luminance', fontsize = 5)
    ax4.set_ylabel('Variance', fontsize = 5)
    ax4.tick_params(labelsize = 7)

    ax5 = axs.add_subplot(3,3,5)
    ax5.scatter(flat_ave5, flat_var5, s = 10, marker = '.', color = "red")
    ax5.plot(flat_ave5, flat_ave5 * fit5[0] + fit5[1], linestyle = "dotted")
    ax5.grid(which = "both", linewidth = 0.5, alpha = 0.1)
    try:
        ax5.set_title(f"gain = {gainkey[4]}, slope = {fit5[0]}, r2 = {r2_5}", fontsize = 5)
    except IndexError:
        pass
    ax5.set_xlabel('Luminance', fontsize = 5)
    ax5.set_ylabel('Variance', fontsize = 5)
    ax5.tick_params(labelsize = 7)

    ax6 = axs.add_subplot(3,3,6)
    ax6.scatter(flat_ave6, flat_var6, s = 10, marker = '.', color = "red")
    ax6.plot(flat_ave6, flat_ave6 * fit6[0] + fit6[1], linestyle = "dotted")
    ax6.grid(which = "both", linewidth = 0.5, alpha = 0.1)
    try:
        ax6.set_title(f"gain = {gainkey[5]}, slope = {fit6[0]}, r2 = {r2_6}", fontsize = 5)
    except IndexError:
        pass
    ax6.set_xlabel('Luminance', fontsize = 5)
    ax6.set_ylabel('Variance', fontsize = 5)
    ax6.tick_params(labelsize = 7)

    ax7 = axs.add_subplot(3,3,7)
    ax7.scatter(flat_ave7, flat_var7, s = 10, marker = '.', color = "red")
    ax7.plot(flat_ave7, flat_ave7 * fit7[0] + fit7[1], linestyle = "dotted")
    ax7.grid(which = "both", linewidth = 0.5, alpha = 0.1)
    try:
        ax7.set_title(f"gain = {gainkey[6]}, slope = {fit7[0]}, r2 = {r2_7}", fontsize = 5)
    except IndexError:
        pass
    ax7.set_xlabel('Luminance', fontsize = 5)
    ax7.set_ylabel('Variance', fontsize = 5)
    ax7.tick_params(labelsize = 7)

    ax8 = axs.add_subplot(3,3,8)
    ax8.scatter(flat_ave8, flat_var8, s = 10, marker = '.', color = "red")
    ax8.plot(flat_ave8, flat_ave8 * fit8[0] + fit8[1], linestyle = "dotted")
    ax8.grid(which = "both", linewidth = 0.5, alpha = 0.1)
    try:
        ax8.set_title(f"gain = {gainkey[7]}, slope = {fit8[0]}, r2 = {r2_8}", fontsize = 5)
    except IndexError:
        pass
    ax8.set_xlabel('Luminance', fontsize = 5)
    ax8.set_ylabel('Variance', fontsize = 5)
    ax8.tick_params(labelsize = 7)

    ax9 = axs.add_subplot(3,3,9)
    ax9.scatter(flat_ave9, flat_var9, s = 10, marker = '.', color = "red")
    ax9.plot(flat_ave9, flat_ave9 * fit9[0] + fit9[1], linestyle = "dotted")
    ax9.grid(which = "both", linewidth = 0.5, alpha = 0.1)
    try:
        ax9.set_title(f"gain = {gainkey[8]}, slope = {fit9[0]}, r2 = {r2_9}", fontsize = 5)
    except IndexError:
        pass
    ax9.set_xlabel('Luminance', fontsize = 5)
    ax9.set_ylabel('Variance', fontsize = 5)
    ax9.tick_params(labelsize = 7)

    plt.show()

# Main routine
if __name__ == '__main__':
    # デフォルトパラメータの設定
    DEFAULT_LOAD_FITS_PATH = 'I:\data/*.fit'
    load_fits_path = DEFAULT_LOAD_FITS_PATH

    calculate_flat_method1_Flag = True  # If True, conversion factor measurement method 1 will be used.
    calculate_flat_method2_Flag = False # If True, conversion factor measurement method 2 will be used.
        
    # Check for the existence of fits files
    if(len(glob.glob(load_fits_path)) == 0):
        print(f"Exit the program because the fits file does not exist. => {load_fits_path}")
        exit()

    # Load start time
    startTime = time.time()

    # Load Fits and Store data
    array_x, array_y = getSenserPixelSize(load_fits_path)
    stack, stack1, stack2, stack3, stack4, stack5, stack6, stack7, stack8, stack9, gainstack, fitsNum, gainkey, gainvalue = loadFilesStore3DnumpyArray(load_fits_path, array_x, array_y)
    print(f'{np.shape(stack)[0]} files loaded.')
    print(f'Loading complete. Elapsed time ={time.time() - startTime} s.')

    if(calculate_flat_method1_Flag):
        del stack
        gc.collect()
        flat_ave1, flat_var1, fit1, r2_1, flat_ave2, flat_var2, fit2, r2_2, flat_ave3, flat_var3, fit3, r2_3, flat_ave4, flat_var4, fit4, r2_4, flat_ave5, flat_var5, fit5, r2_5, flat_ave6, flat_var6, fit6, r2_6, flat_ave7, flat_var7, fit7, r2_7, flat_ave8, flat_var8, fit8, r2_8, flat_ave9, flat_var9, fit9, r2_9 = calculate_flat_1(stack1, stack2, stack3, stack4, stack5, stack6, stack7, stack8, stack9, fitsNum, gainvalue)
        plot_flat(flat_ave1, flat_var1, fit1, r2_1, flat_ave2, flat_var2, fit2, r2_2, flat_ave3, flat_var3, fit3, r2_3, flat_ave4, flat_var4, fit4, r2_4, flat_ave5, flat_var5, fit5, r2_5, flat_ave6, flat_var6, fit6, r2_6, flat_ave7, flat_var7, fit7, r2_7, flat_ave8, flat_var8, fit8, r2_8, flat_ave9, flat_var9, fit9, r2_9, gainkey)

    if(calculate_flat_method2_Flag):
        del stack
        gc.collect()
        flat_ave1, flat_var1, fit1, r2_1, flat_ave2, flat_var2, fit2, r2_2, flat_ave3, flat_var3, fit3, r2_3, flat_ave4, flat_var4, fit4, r2_4, flat_ave5, flat_var5, fit5, r2_5, flat_ave6, flat_var6, fit6, r2_6, flat_ave7, flat_var7, fit7, r2_7, flat_ave8, flat_var8, fit8, r2_8, flat_ave9, flat_var9, fit9, r2_9 = calculate_flat_2(stack1, stack2, stack3, stack4, stack5, stack6, stack7, stack8, stack9, fitsNum, gainvalue)
        plot_flat(flat_ave1, flat_var1, fit1, r2_1, flat_ave2, flat_var2, fit2, r2_2, flat_ave3, flat_var3, fit3, r2_3, flat_ave4, flat_var4, fit4, r2_4, flat_ave5, flat_var5, fit5, r2_5, flat_ave6, flat_var6, fit6, r2_6, flat_ave7, flat_var7, fit7, r2_7, flat_ave8, flat_var8, fit8, r2_8, flat_ave9, flat_var9, fit9, r2_9, gainkey)

print(f'Processing completed. Elapsed time ={time.time() - startTime} s.')
