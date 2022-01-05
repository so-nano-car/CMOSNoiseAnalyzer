"""
===========================================
CMOS camera noise analysis program ver.1.1
===========================================
*By: so-nano-car inspired by Mr. "Apranat"*
*Original source code URL:https://so-nano-car.com/noise-evaluation-of-qhy5iii174m
*Arrange: Daitoshi 
*License: BSD*
*プログラムの使用方法*
コマンドライン引数を指定しない場合は、実行されたカレントディレクトリのサブディレクトリ./fits/ 以下の*.fitsファイルを処理対象とします
一つ目の引数は、fitsファイルのパスとして解釈される パスの最後の文字は \ or / では無い状態で指定すること 例 c:\hoge\fits
二つ目の引数は、グラフのタイトルとして解釈される 空白文字は入れずに、_ で代用すること 例 ASI294_Dark_gain300_30c*128files
コマンドライン例 
python fitsAnalayze.py "C:\home\SharpCap Captures\2021-09-26\Capture\Dark" ASI294_Dark_gain300_30c*128files

*Revision history*
ver 1.0 New release
ver 1.1 arrange by Daitoshi, add new feature (Time-series analysis)
ver 1.2 improve processing speed, add new feature (image integration) 6-Oct.2021
ver 1.3 add new feature (conversion factor measurement) 5-Jan.2022
"""

import matplotlib.pyplot as plt
import numpy as np
import glob
from astropy.visualization import astropy_mpl_style
from astropy.io import fits
import time
import sys
from numpy.lib import savetxt
from multiprocessing import Pool
from numpy.lib.function_base import median

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
            gainstack[i,:] = gain
            print(f"loadFitsFiles {i+1}/{fitsNum}")
    return (stack, gainstack, fitsNum)

# Load offset values of each frame
def loadOffsetValue(fitspath):
    fitsFiles = glob.glob (fitspath)
    fitsNum   = len(fitsFiles)
    hdrstack = np.empty((fitsNum,1), dtype = np.int16)
    for i, img in enumerate(fitsFiles):
            hdul = fits.open(img)
            hdr = hdul[0].header['OFFSET']
            hdrstack[i,:] = hdr
    return hdrstack

# Calculate average lightness and variance for each flat frame
def calculate_flat(stack, fitsNum):
    stack14 = stack / 4
    flat_ave = np.mean(np.mean(stack14, axis = 1), axis = 1)
    flat_var = np.zeros((fitsNum, 1))
    for i in range (fitsNum):
        flat_var[i,:] = np.var(stack14[i,:,:])
    print(f'flat_var = {flat_var}')
    fit = np.polyfit(flat_ave, flat_var, 1)
    
    return (flat_ave, flat_var, fit)

def plot_flat(flat_ave, flat_var, gainstack, fit):
    y_fit = fit[0] * flat_ave + fit[1]
    
    plt.scatter(flat_ave, flat_var, s = 7.0)
    plt.plot(flat_ave, y_fit)
    plt.title(f'gain = {gainstack[0]}, y = {fit[0]} * x + {fit[1]}')
    
    plt.show()

# Calculate x,y
def calculatePlottingPoint(stack, array_x, array_y):
    # Calculate median and std.dev. of each pixel
    median = np.median(stack, axis = 0)
    stddev = np.std(stack, axis = 0, ddof = 0, dtype = np.float32)
    
    # Reshape median and std.dev. array for plotting
    x = median.reshape([array_y * array_x,1])
    y = stddev.reshape([array_y * array_x,1])
    return(x,y)

# Calculate median and std.dev. of each pixel row
def calculate(stack):
    median1 = np.median(stack, axis = 0)
    stddev1 = np.std(stack, axis = 0,ddof = 0, dtype = np.float32)
    return(median1,stddev1)

# Integrate image (average, no pixel rejection)
def integrateImage(stack):
    integrated = np.mean(stack, axis = 0, dtype = np.float32)
    return(integrated)

# Export fits file
def exportFits(integrated):
    hdu = fits.PrimaryHDU(integrated)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto('processed.fits',overwrite=True)

# Display fits file
def displayFits(integrated):
    plt.figure()
    plt.imshow(integrated, cmap='gray')
    plt.suptitle('Preview')
    plt.colorbar()
    plt.show()

# Export to csv file
def exportToCsvFile(x,y,saveCsvname):
    data = np.concatenate([x, y], 1)
    np.savetxt(saveCsvname, data, delimiter = ',')

# Plot scatter only 
def plotResults_scatter(x, y, title):
    plt.style.use(astropy_mpl_style)
    plt.scatter(x, y, s = 0.3, marker = '.', color = "blue")
    plt.grid(which = "both", linewidth = 0.5, alpha = 0.1)
    plt.suptitle(title)
    plt.xlabel('Median')
    plt.ylabel('Std.dev.')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim([100,100_000])
    plt.ylim([1,100_000])
    plt.show()

# Time-series analysis: extract pixel(s) that matches the criteria
def ExtractPixels(median, stddev, stack):
    median_low = 200
    median_high = 400
    stddev_low = 300
    stddev_high = 1000
    area_x = [median_low,median_high,median_high,median_low]
    area_y = [stddev_low,stddev_low,stddev_high,stddev_high]
    index = np.where((median >median_low) & (median < median_high) & (stddev > stddev_low) & (stddev < stddev_high))
    index_y = index[0]
    index_x = index[1]
    print("{0} pixel(s) matched the criteria: {1} < median < {2}, {3} < stddev < {4}, showing first 8 pixels:".format(len(index_y),median_low,median_high,stddev_low,stddev_high))
    i = 0 # If i = 0, first 8 pixels are analyzed.
    pixel1 = stack[:,index_y[i],index_x[i]]
    pixel2 = stack[:,index_y[i+1],index_x[i+1]]
    pixel3 = stack[:,index_y[i+2],index_x[i+2]]
    pixel4 = stack[:,index_y[i+3],index_x[i+3]]
    pixel5 = stack[:,index_y[i+4],index_x[i+4]]
    pixel6 = stack[:,index_y[i+5],index_x[i+5]]
    pixel7 = stack[:,index_y[i+6],index_x[i+6]]
    pixel8 = stack[:,index_y[i+7],index_x[i+7]]
    return(i, pixel1, pixel2, pixel3, pixel4, pixel5, pixel6, pixel7, pixel8, index_x, index_y, area_x, area_y)

# Plot time-series analysis
def plotResults_series(i, x, y, pixel1, pixel2, pixel3, pixel4, pixel5, pixel6, pixel7, pixel8, index_x, index_y, area_x, area_y):
    y_lim = 70000 # y axis limit
    j = i

    axs = plt.figure(figsize=(10,10))
    ax1 = axs.add_subplot(3,3,1)
    ax1.scatter(x,y,s = 0.3, marker = '.', color = "red")
    ax1.grid(which = "both", linewidth = 0.5, alpha = 0.1)
    ax1.set_title("Scatter")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim([1,100000])
    ax1.set_ylim([1,100000])
    ax1.fill(area_x,area_y,color = "blue", alpha = 0.5)
    ax1.tick_params(labelsize = 7)
    
    ax2 = axs.add_subplot(3,3,2)
    ax2.plot(pixel1, color = "black")
    ax2.set_title("Pixel ({0},{1})".format(index_x[j],index_y[j]))
    ax2.set_ylim([0,y_lim])
    ax2.tick_params(labelsize = 7)

    ax3 = axs.add_subplot(3,3,3)
    ax3.plot(pixel2, color = "black")
    ax3.set_title("Pixel ({0},{1})".format(index_x[j+1],index_y[j+1]))
    ax3.set_ylim([0,y_lim])
    ax3.tick_params(labelsize = 7)

    ax4 = axs.add_subplot(3,3,4)
    ax4.plot(pixel3, color = "black")
    ax4.set_title("Pixel ({0},{1})".format(index_x[j+2],index_y[j+2]))
    ax4.set_ylim([0,y_lim])
    ax4.tick_params(labelsize = 7)

    ax5 = axs.add_subplot(3,3,5)
    ax5.plot(pixel4, color = "black")
    ax5.set_title("Pixel ({0},{1})".format(index_x[j+3],index_y[j+3]))
    ax5.set_ylim([0,y_lim])
    ax5.tick_params(labelsize = 7)

    ax6 = axs.add_subplot(3,3,6)
    ax6.plot(pixel5, color = "black")
    ax6.set_title("Pixel ({0},{1})".format(index_x[j+4],index_y[j+4]))
    ax6.set_ylim([0,y_lim])
    ax6.tick_params(labelsize = 7)

    ax7 = axs.add_subplot(3,3,7)
    ax7.plot(pixel6, color = "black")
    ax7.set_title("Pixel ({0},{1})".format(index_x[j+5],index_y[j+5]))
    ax7.set_ylim([0,y_lim])
    ax7.tick_params(labelsize = 7)

    ax8 = axs.add_subplot(3,3,8)
    ax8.plot(pixel7, color = "black")
    ax8.set_title("Pixel ({0},{1})".format(index_x[j+6],index_y[j+6]))
    ax8.set_ylim([0,y_lim])
    ax8.tick_params(labelsize = 7)

    ax9 = axs.add_subplot(3,3,9)
    ax9.plot(pixel8, color = "black")
    ax9.set_title("Pixel ({0},{1})".format(index_x[j+7],index_y[j+7]))
    ax9.set_ylim([0,y_lim])
    ax9.tick_params(labelsize = 7)
    plt.show()

# Calculate max, average, min of whole pixels (used for offset value determination)
def calculateOffsetStats(stack):
    wMax = np.amax(np.amax(stack,axis = 1), axis = 1)
    wAve = np.average(np.average(stack, axis = 1), axis = 1)
    wMin = np.min(np.min(stack, axis = 1), axis = 1)
    return(wMax,wAve,wMin)

def plotOffsetStats(hdrstack, wMax, wAve, wMin):
    x = hdrstack

    axs = plt.figure(figsize=(10,10))

    ax1 = axs.add_subplot(1,3,1)
    ax1.scatter(x, wMax, s = 1, marker = '.', color = "red")
    ax1.grid(which = "both", linewidth = 0.5, alpha = 0.1)
    ax1.set_title("Max")
    #ax1.set_xlim([0,30])
    #ax1.set_ylim([0,3000])
    ax1.set_xlabel('Offset')
    ax1.set_ylabel('Luminance')
    ax1.tick_params(labelsize = 7)

    ax2 = axs.add_subplot(1,3,2)
    ax2.scatter(x, wAve, s = 1, marker = '.', color = "red")
    ax2.grid(which = "both", linewidth = 0.5, alpha = 0.1)
    ax2.set_title("Ave")
    #ax2.set_xlim([0,30])
    #ax2.set_ylim([0,3000])
    ax2.set_xlabel('Offset')
    ax2.tick_params(labelsize = 7)

    ax3 = axs.add_subplot(1,3,3)
    ax3.scatter(x, wMin, s = 1, marker = '.', color = "red")
    ax3.grid(which = "both", linewidth = 0.5, alpha = 0.1)
    ax3.set_title("Min")
    #ax3.set_xlim([3,5])
    #ax3.set_ylim([0,200])
    ax3.set_xlabel('Offset')
    ax3.tick_params(labelsize = 7)
    plt.show()

# Main routine
if __name__ == '__main__':
    # デフォルトパラメータの設定
    DEFAULT_LOAD_FITS_PATH = 'I:\data/*.fit'
    DEFAULT_SAVE_CSV_NAME  = './test/result.csv'
    DEFAULT_FITS_NAME = './processed.fits'
    DEFAULT_TITLE          = "Dark(120s Gain100)"
    load_fits_path = DEFAULT_LOAD_FITS_PATH
    save_csv_path  = DEFAULT_SAVE_CSV_NAME
    save_fits_path = DEFAULT_FITS_NAME
    title = DEFAULT_TITLE

    CsvExportFlag = False # If CsvExportFlag is True, output a csv file.
    seriesAnalysisFlag = False # If True, output time-series analysis result. (calculateScatterFlag must be True)
    calculateScatterFlag = False # If True, median and std. dev. of each pixel will be calculated.
    plotScatterFlag = False # If True, only scatter plot will be generated.
    integrateFlag = False #If True, image integration will be carried out.
    exportFitsFlag = False #If True, fits file of integrated image will be generated.
    offsetStatsFlag = False # If True, output offset-luminance analysis result.
    calculate_flat_Flag = True # If True, conversion factor measurement will be carried out.
    
    # コマンドライン引数の処理
    args = sys.argv
    if(2 <= len(args)):
        load_fits_path = args[1] + '/*.fit'
    if(3 <= len(args)):
        title = args[2]

    # Check for the existence of fits files
    if(len(glob.glob(load_fits_path)) == 0):
        print(f"Exit the program because the fits file does not exist. => {load_fits_path}")
        exit()

    # Load start time
    startTime = time.time()

    # Load Fits and Store data
    array_x, array_y = getSenserPixelSize(load_fits_path)
    stack, gainstack, fitsNum = loadFilesStore3DnumpyArray(load_fits_path, array_x, array_y)
    print(f'{np.shape(stack)[0]} files loaded.')
    print(f'Loading complete. Elapsed time ={time.time() - startTime} s.')

    # Plot offset statistics
    if(offsetStatsFlag):
        hdrstack = loadOffsetValue(load_fits_path)
        wMax, wAve, wMin = calculateOffsetStats(stack)
        print(f'Processing completed. Elapsed time ={time.time() - startTime} s.')
        plotOffsetStats(hdrstack, wMax, wAve, wMin)
        
    # Calculate Plotting Point
    if(calculateScatterFlag):
        x,y = calculatePlottingPoint(stack, array_x, array_y)
        print(f'Processing completed. Elapsed time ={time.time() - startTime} s.')
    
    # CsvExport
    if(CsvExportFlag):
        exportToCsvFile(x,y, save_csv_path)
        print("Csv file exported to {0}".format(save_csv_path))

    # Plot scatter
    if(plotScatterFlag):
        plotResults_scatter(x,y,title)
       
    # Time-series analysis
    if(seriesAnalysisFlag):
        median1, stddev1 = calculate(stack)
        x, y = calculatePlottingPoint(stack, array_x, array_y)
        i, pixel1, pixel2, pixel3, pixel4, pixel5, pixel6, pixel7, pixel8, index_x, index_y, area_x, area_y = ExtractPixels(median1,stddev1,stack)
        print(f'Processing completed. Elapsed time ={time.time() - startTime} s.')
        plotResults_series(i, x, y, pixel1, pixel2, pixel3, pixel4, pixel5, pixel6, pixel7, pixel8, index_x, index_y, area_x, area_y) 
    
    # Image integration
    if(integrateFlag):
        integrated = integrateImage(stack)
        displayFits(integrated)
        print(f'Integration completed. Elapsed time ={time.time() - startTime} s.')
    
    if(exportFitsFlag):
        exportFits(integrated)
        print('Fits image exported to: {0}'.format(save_fits_path))
    
    if(calculate_flat_Flag):
        flat_ave, flat_var, fit = calculate_flat(stack, fitsNum)
        plot_flat(flat_ave, flat_var, gainstack, fit)


