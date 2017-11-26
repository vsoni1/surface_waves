import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import path
from scipy.ndimage.filters import gaussian_filter
from PIL import Image, ImageDraw
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.feature import canny
import pims

class Click_points:
    """ Class for getting points from figure"""
    def __init__(self, fig):
        self.figure = fig 
        self.xs = []
        self.ys = []
        self.cid = self.figure.canvas.mpl_connect(
            'button_press_event', self
            )
        
    def __call__(self, event):
        print('xdata = %f, ydata= %f' %(event.xdata, event.ydata))
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        
def get_points(F):
    """ Returns points in image obtained form user clicks.
    
    Parameters
    ----------
    F : array
        2D image where points are to be found
        
    Returns
    -------
    array
        Array of interger (x,y) coordinates of points that were clicked
    
    """
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    plt.imshow(F, origin='lower')
    points = Click_points(fig)
    plt.show()
    
    return np.array([points.xs,points.ys]).T
        
def extract_profile(image, gaussian_width=5, sigma=.2):
    """extract profile of bright region in image
    
    Parameters
    ----------
    image : array
        2D image where outline profile to be found
    gaussian_width : array
        2D image where outline profile to be found
    sigma : array
        2D image where outline profile to be found
    
    Returns
    -------
    array
        Array of interger (x,y) coordinates of the outline points
    
    """
    image_filtered = gaussian_filter(
        image, gaussian_width
        )
    thresh_otsu_filtered = threshold_otsu(image_filtered) 
    binary = image_filtered > thresh_otsu_filtered
    edges = canny(binary, sigma=sigma)
    coords = np.argwhere(edges == True)
    
    return coords 
    
def polygon_fill(framesizex, framesizey, x, y, fill=0, background=1):
    """Fills inside and outside of polygon with provided values.
    
    Parameters
    ----------
    x : array
        x-coordinates of polygon
    y : array
        y-coordinates of polygon
    framesizex : array
        x-dimension of output image
    framesizey : array
        y-dimension of output image.
    fill : float
        value to assign inside polygon
    background : float
        value to assign outside polygon
    
    Returns
    -------
    array
        2D image of polygon
    
    """
    polygon = zip(x, y)
    img = Image.new('I', (framesizex, framesizey), background)
    ImageDraw.Draw(img).polygon(polygon, outline=1, fill=fill)
    return np.array(img)
    
def mask_frame(image):
    """Makes mask for image
    
    Parameters
    ----------
    x : array
        2D image to be make mask for
    
    Returns
    -------
    array
        2D mask image
    
    """
    b = get_points(image)
    mask = polygon_fill(
        framesizex=np.shape(image)[1],
        framesizey=np.shape(image)[0],
        x=b[:,0], y=b[:,1], fill=1, background=0,
        )
    return np.array(mask)
    
def dispersion(data, T1=1, T2=.5):
    """plots power spectrum for the waves
    
    Parameters
    ----------
    data : array
        2D (t dim by x dim) height profile data
        each row is an instantaneous height profile
    T1 : float
        pixel conversion for time dimension
    T2 : array
        pixel conversion for space
    
    """
    h = data - np.tile(np.mean(data, axis=0), (np.shape(data)[0], 1))
    h_f = np.fft.fftshift(np.fft.fft2(h))
    h_p = np.abs(h_f) * np.abs(h_f)
    d = h_p / np.tile(np.mean(h_p, axis=0), (np.shape(h_p)[0],1))
    N1, N2 = np.shape(data)
    
    fig = plt.figure(figsize=(15, 15), facecolor='w')
    ax = fig.add_subplot(111)
    ax.imshow(
        np.log(1 + d),
        extent=(-1 / T2 / 2.0, 1 / T2 / 2.0, -1 / T1 / 2.0, 1 / T1 / 2.0),
        interpolation='nearest'
        )
    # ax.set_xlim([-.15, .15])
    # ax.set_ylim([-.15, .15])
    ax.set_xlabel('k', fontsize=20)
    ax.set_ylabel('$\omega$', fontsize=20)
    plt.show()
    plt.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='track waves on surface of bright region in movie'
        )
    parser.add_argument(
        'input', help='path to directory holding frames of movie'
        )
    parser.add_argument(
        '--start', type=int, default=0, help='first frame to analyze'
        )
    parser.add_argument(
        '--end', type=int, default=None, help='last frame to analyze'
        )
    parser.add_argument(
        '--extension', type=str, default='jpg', help='file extension of frames'
        )
    args = parser.parse_args()
    moviepath = args.input
    startframe = args.start
    endframe = args.end
    extension = args.extension
    
    if not os.path.exists(moviepath + '/h_analysis'):
        os.makedirs(moviepath + '/h_analysis')
        
    h = []
    frames = pims.ImageSequence(moviepath + '/*.' + extension)
    image = rgb2gray(frames[startframe])
    mask = mask_frame(image)
    roi = path.Path(
        get_points(image * mask))
    startx, starty = get_points(image * mask)[0]
    endx, endy = get_points(image * mask)[0]
    xn = np.arange(startx, endx)
    meany = np.mean([endy, starty])
    
    coords = extract_profile(
        image * mask, gaussian_width=5, sigma=.2
        )
    plt.imshow(image * mask, cmap='gray')
    plt.plot(coords[:,1],coords[:,0], 'or')
    plt.plot([startx, endx], [meany, meany])
    plt.show()
    
    for ind,item in enumerate(frames[startframe:endframe]):
        print(ind)
        image = rgb2gray(item)
        coords = extract_profile(
            image * mask, gaussian_width=5, sigma=.2
            )
        y = coords[:, 0]
        x = coords[:, 1]
        inside = roi.contains_points(np.array([x, y]).T)
        x = x[inside]
        y = y[inside]
        xind = np.array([np.where(x==xi) for xi in set(x)])
        xd = list(set(x))
        yd = [np.min(np.abs(y[xi[0]] - meany)) for xi in xind]
        yn = np.interp(xn, xd, np.abs(yd - meany))
        h.append(yn)
        # np.save(moviepath + '/h_analysis/height.npy', h)
        
    dispersion(h, T1=1, T2=.5)