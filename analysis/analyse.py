import ntpath
# import cv2
import sys
import scipy
import scipy.misc
import imageio
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

# http://www.scipy-lectures.org/advanced/image_processing/
def threshold( img, limit ):
    img2 = img.copy()
    siz = img2.shape
    for y in range(siz[1]):
        for x in range(siz[0]):
            v = img[x,y]
            if ( v < limit ):
                img2[x,y] = 255
            else:
                img2[x,y] = 0
    return img2

def bitplane( img, bit ):
    # TODO: return img & (1<<bit)

    img2 = img.copy()
    siz = img2.shape
    for y in range(siz[1]):
        for x in range(siz[0]):
            v = img[x,y]
            if ( (v & (1<<bit)) != 0 ):
                img2[x,y] = 255
            else:
                img2[x,y] = 0
    return img2

def test_bitplane_gen( img, bit ):
    img2 = img.copy()
    siz = img2.shape
    for y in range(siz[1]):
        for x in range(siz[0]):
            img2[x,y] = np.uint8(1<<bit)
    return img2

def singlechannel_binary_to_rgb( img ):
    ret = np.zeros( (img.shape[0], img.shape[1],3) )
    ret[img!=0] = [1,1,1]
    return ret

# Get PSD 1D (total power spectrum by angular bin)
# https://medium.com/tangibit-studios/2d-spectrum-characterization-e288f255cc59
def GetRPSD(psd2D, dTheta, rMin, rMax):
    h  = psd2D.shape[0]
    w  = psd2D.shape[1]
    wc = w//2
    hc = h//2

    # note that displaying PSD as image inverts Y axis
    # create an array of integer angular slices of dTheta
    Y, X  = np.ogrid[0:h, 0:w]
    theta = np.rad2deg(np.arctan2(-(Y-hc), (X-wc)))
    theta = np.mod(theta + dTheta/2 + 360, 360)
    theta = dTheta * (theta//dTheta)
    theta = theta.astype(np.int)

    # mask below rMin and above rMax by setting to -100
    R     = np.hypot(-(Y-hc), (X-wc))
    mask  = np.logical_and(R > rMin, R < rMax)
    theta = theta + 100
    theta = np.multiply(mask, theta)
    theta = theta - 100
    
    # SUM all psd2D pixels with label 'theta' for 0<=theta❤60 between rMin and rMax
    angF  = np.arange(0, 360, int(dTheta))
    psd1D = ndimage.sum(psd2D, theta, index=angF)

    # normalize each sector to the total sector power
    pwrTotal = np.sum(psd1D)
    psd1D    = psd1D/pwrTotal

    return angF, psd1D


def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile 

def fftmag( img ):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return np.log(np.abs(fshift))


if ( len( sys.argv ) ) < 2:
    sys.exit("usage: analyse.py <filename.bmp>")

#########
# init
pt = sys.argv[1]
fn = path_leaf( pt )

# img = scipy.misc.imread( sys.argv[1] )
# img = scipy.misc.imread( sys.argv[1], mode='L' )
img0 = imageio.imread( pt )
img = img0[:,:,1]

# note: testing result of adding two rpdf bluenoise texture-channels to form a tpdf
# img =  0.5 * img0[:,:,1] + 0.5 * img0[:,:,2]

###########
# calc
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
# # magnitude_spectrum = 20*np.log(np.abs(fshift))
# magnitude_spectrum = np.log(np.abs(fshift))
phase_spectrum = np.angle(fshift)
magnitude_spectrum = fftmag(img)

fft_radialprofile = radial_profile( magnitude_spectrum, (0.5*magnitude_spectrum.shape[0], 0.5*magnitude_spectrum.shape[1]) )

angF, psd1D = GetRPSD(magnitude_spectrum, 5, 1, 64)

###########
# plotting

interp = 'bicubic' #nearest

# plt.clf()

SMALL_SIZE = 4
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title


fig, axs = plt.subplots(4,4)

axs[0,0].imshow(img, cmap = 'gray', interpolation=interp)
axs[0,0].set( title='Input Image channel=0/' + str(img0.shape[2]) + '\n' + fn ), plt.xticks([]), plt.yticks([])
axs[0,0].axis('off')

axs[0,1].hist(img.ravel(), bins=256, range=(0,255))
axs[0,1].set(title="histogram")

axs[0,2].axis('off')
axs[0,3].axis('off')

axs[1,0].imshow(magnitude_spectrum, cmap = 'gray', interpolation=interp)
axs[1,0].set( title='FFT Magnitude Spectrum')
axs[1,0].axis('off')

axs[1,1].plot(fft_radialprofile)
axs[1,1].set( title='FFT Radial profile')
axs[1,1].set_ylim([0,13])

# TODO: polar 
axs[1,2].plot(angF, psd1D)
axs[1,2].set( title='FFT Radial Symmetry profile')
axs[1,2].set_ylim([0,0.1])
axs[1,2].grid()

axs[1,3].imshow(phase_spectrum, cmap='gray', interpolation=interp)
axs[1,3].set(title='FFT Phase Spectrum')
axs[1,3].axis('off')

# reference: https://blog.demofox.org/2018/08/12/not-all-blue-noise-is-created-equal/ 
# TODO: better layout ½
img_threshold2 = threshold(img, 2)
img_threshold4 = threshold(img, 4)
img_threshold8 = threshold(img, 8)
img_threshold16 = threshold(img, 16)

axs[2,0].imshow(img_threshold2, cmap = 'gray', interpolation=interp)
axs[2,0].set( title='Thresholded 2/255')
axs[2,0].axis('off')
 
axs[2,1].imshow(img_threshold4, cmap = 'gray', interpolation=interp)
axs[2,1].set(title='Thresholded 4/255')
axs[2,1].axis('off')

axs[2,2].imshow(img_threshold8, cmap = 'gray', interpolation=interp)
axs[2,2].set(title='Thresholded 8/255')
axs[2,2].axis('off')
 
axs[2,3].imshow(img_threshold16, cmap = 'gray', interpolation=interp)
axs[2,3].set( title='Thresholded 16/255')
axs[2,3].axis('off')


axs[3,0].imshow(fftmag(img_threshold2), cmap = 'gray', interpolation=interp)
axs[3,0].set(title='Magnitude Spectrum')
axs[3,0].axis('off')

axs[3,1].imshow(fftmag(img_threshold4), cmap = 'gray', interpolation=interp)
axs[3,1].set(title='Magnitude Spectrum')
axs[3,1].axis('off')

axs[3,2].imshow(fftmag(img_threshold8), cmap = 'gray', interpolation=interp)
axs[3,2].set(title='Magnitude Spectrum')
axs[3,2].axis('off')

axs[3,3].imshow(fftmag(img_threshold16), cmap = 'gray', interpolation=interp)
axs[3,3].set(title='Magnitude Spectrum')
axs[3,3].axis('off')


# img_threshold242 = threshold(img, 242)
# img_threshold246 = threshold(img, 246)
# img_threshold250 = threshold(img, 250)
# img_threshold254 = threshold(img, 254)

# axs[4,0].imshow(img_threshold242, cmap = 'gray', interpolation=interp)
# axs[4,0].set( title='Thresholded 242/255')
# axs[4,0].axis('off')
#  
# axs[4,1].imshow(img_threshold246, cmap = 'gray', interpolation=interp)
# axs[4,1].set(title='Thresholded 246/255')
# axs[4,1].axis('off')
# 
# axs[4,2].imshow(img_threshold250, cmap = 'gray', interpolation=interp)
# axs[4,2].set(title='Thresholded 250/255')
# axs[4,2].axis('off')
#  
# axs[4,3].imshow(img_threshold254, cmap = 'gray', interpolation=interp)
# axs[4,3].set( title='Thresholded 254/255')
# axs[4,3].axis('off')
# 
# 
# axs[5,0].imshow(fftmag(img_threshold242), cmap = 'gray', interpolation=interp)
# axs[5,0].set(title='Magnitude Spectrum')
# axs[5,0].axis('off')
# 
# axs[5,1].imshow(fftmag(img_threshold246), cmap = 'gray', interpolation=interp)
# axs[5,1].set(title='Magnitude Spectrum')
# axs[5,1].axis('off')
# 
# axs[5,2].imshow(fftmag(img_threshold250), cmap = 'gray', interpolation=interp)
# axs[5,2].set(title='Magnitude Spectrum')
# axs[5,2].axis('off')
# 
# axs[5,3].imshow(fftmag(img_threshold254), cmap = 'gray', interpolation=interp)
# axs[5,3].set(title='Magnitude Spectrum')
# axs[5,3].axis('off')


# TODO: individual bitplanes
# axs[2,0].imshow( singlechannel_binary_to_rgb(bitplane(img,0)), interpolation=interp)
# axs[2,0].set( title='bitplane0')
# axs[2,0].axis('off')
# 
# axs[2,1].imshow( singlechannel_binary_to_rgb(bitplane(img,1)), interpolation=interp)
# axs[2,1].set( title='bitplane1')
# axs[2,1].axis('off')
# 
# axs[2,2].imshow( singlechannel_binary_to_rgb(bitplane(img,2)), interpolation=interp)
# axs[2,2].set( title='bitplane2')
# axs[2,2].axis('off')
# 
# axs[2,3].imshow( singlechannel_binary_to_rgb(bitplane(img,3)), interpolation=interp)
# axs[2,3].set( title='bitplane3')
# axs[2,3].axis('off')


plt.tight_layout()

fn = fn.replace(".", "_")
plt.savefig( 'figure_' + fn + '.png', dpi=300, quality=100, bbox_inches="tight" )
