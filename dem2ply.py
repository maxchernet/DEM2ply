##################
# Convert DEM to PLY
# Example of usage:
# python dem2ply.py --dem output_VX02000379_00101-01701/c2-DEM.tif --tex output_VX02000379_00101-01701/c2-DRG.tif --out ply_VX02000379_01.ply
##################

import gdal
import argparse
import numpy as np
from scipy import interpolate

import matplotlib.pyplot as plt


def interp(arr_2d_1, arr_2d_2):
    """
    Interpolate array arr_2d_1 to the size of arr_2d_2

    :param arr_2d_1: array
    :param arr_2d_2: array
    :return: array
    """

    x = np.arange(0, arr_2d_1.shape[1])
    y = np.arange(0, arr_2d_1.shape[0])
    f = interpolate.interp2d(x, y, arr_2d_1, kind='cubic')

    step_x = arr_2d_1.shape[1] / np.double(arr_2d_2.shape[1])
    step_y = arr_2d_1.shape[0] / np.double(arr_2d_2.shape[0])

    xnew = np.arange(0, arr_2d_1.shape[1], step_x)
    ynew = np.arange(0, arr_2d_1.shape[0], step_y)
    arr_new = f(xnew, ynew)

    return arr_new


def contrast(img_vec_in, shape_2d, do_plot=False):
    """
    Automaticaly improve contrast of img_vec_in

    :param img_vec_in: vector
    :param shape_2d: shape
    :param do_plot: bool
    :return: vector
    """

    img = np.resize(img_vec_in, shape_2d).astype(int)
    hist, bins = np.histogram(img_vec_in, 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    img2 = cdf[img]

    if do_plot:
        # PLot hist
        plt.figure(figsize=(15, 10))
        plt.subplot(121)
        plt.hist(cdf_m, 256, color='r')

        plt.subplot(122)
        plt.imshow(np.resize(img2, img_drg.shape), cmap=plt.get_cmap("Greys"))
        plt.colorbar(fraction=0.2)
        plt.show()
        # end plot hist

    return img2.flatten()



def get_texture(img_drg, do_contrast = 'False'):

    print 'Texture max/min:', img_drg.max(), img_drg.min()

    img_tex_vec = img_drg.flatten()
    img_tex_vec[img_tex_vec < 0] = 0
    img_tex_vec[img_tex_vec > 255] = 0
    print 'Texture max/min:', img_tex_vec.max(), img_tex_vec.min()
    #if img_tex_vec.max() != 255:
    img_tex_vec = abs((img_tex_vec / img_tex_vec.max()) * 255).astype(int)
    print 'Texture max/min:', img_tex_vec.max(), img_tex_vec.min()

    if do_contrast == 'True':
        img_tex_vec = contrast(img_tex_vec, img_drg.shape)

    return img_tex_vec



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='convert DEM to ply')

    parser.add_argument('--dem', dest='dem', help='input DEM')
    parser.add_argument('--tex', dest='tex', help='input texture')
    parser.add_argument('--out', dest='out', help='Output ply file')
    parser.add_argument('--ce', dest='ce', help='Contrast enchansment', default='False' )

    args = parser.parse_args()

    if args.dem == None:
        print 'No DEM files provided.\n Usage: dem2ply.py --dem dem_file --tex texture_file --out output_ply'
        exit()

    if args.out == None:
        print 'No DEM output file provided.\n Usage: dem2ply.py --dem dem_file --tex texture_file --out output_ply'
        exit()

    gds = gdal.Open(args.dem)
    img_dem = gds.GetRasterBand(1).ReadAsArray()
    img_dem[img_dem < -500] = 0
    img_vec = img_dem.flatten()
    ind = img_vec > -500
    img_vec = img_vec[ind]

    img_tex_vec = {}

    if args.tex != None:
        gds = gdal.Open(args.tex)
        band_range = [1, 1, 1]
        band_names = ['red', 'green', 'blue']
        if gds.RasterCount == 3:
            band_range = [1, 2, 3]
        # img_drg = np.zeros((gds.RasterYSize, gds.RasterXSize, 3))
        for i, b in enumerate(band_range):
            img_drg = gds.GetRasterBand(b).ReadAsArray()
            if np.logical_or(img_drg.shape[0] != img_dem.shape[0], img_drg.shape[1] != img_dem.shape[1]):
                img_drg = interp(img_drg, img_dem)
            img_tex_vec[band_names[i]] = get_texture(img_drg, args.ce)
            img_tex_vec[band_names[i]] = img_tex_vec[band_names[i]][ind]
    else:
        for bn in band_names:
            img_tex_vec[bn] = img_vec.copy()

    yv, xv = np.meshgrid(np.arange(img_dem.shape[1]), np.arange(img_dem.shape[0]))

    xv = xv.flatten()[ind]
    yv = yv.flatten()[ind]

    img_data = np.zeros((xv.flatten().shape[0], 6))
    img_data[:, 0] = xv
    img_data[:, 1] = yv
    img_data[:, 2] = img_vec
    img_data[:, 3] = img_tex_vec['red']
    img_data[:, 4] = img_tex_vec['green']
    img_data[:, 5] = img_tex_vec['blue']

    head_str = 'ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\n' % xv.shape[0] +\
    'property float y\nproperty float z\nproperty uchar red\nproperty uchar green\n' +\
    'property uchar blue\nend_header'

    np.savetxt(args.out, img_data, header=head_str, fmt='%.4f %.4f %.4f %d %d %d', comments='')
