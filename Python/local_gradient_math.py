"""
Here are implemented all the main functions of the local gradient method

Authors: Oleksandr Perederii, Anatolii Kashchuk
"""

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.fft import fft2, ifft2
from scipy.linalg import lstsq
from scipy.io import loadmat
from scipy import stats
import numpy as np
import math
import cv2
import os
import PIL

from typing import Tuple


def disk_filter(fltSz: float) -> np.ndarray:
    """
    CREATE DISK FILTER
    :param fltSz: float; radius of disk filter
    :return: h; numpy.array of floats; shape:(2*fltSz+1, 2*fltSz+1)
    """
    crad = math.ceil(fltSz - 0.5)
    [y, x] = np.mgrid[-crad: crad + 1, -crad: crad + 1]
    maxxy = np.maximum(abs(x), abs(y))
    minxy = np.minimum(abs(x), abs(y))

    m1 = (fltSz ** 2 < (maxxy + 0.5) ** 2 + (minxy - 0.5) ** 2) * (minxy - 0.5) + \
         (fltSz ** 2 >= (maxxy + 0.5) ** 2 + (minxy - 0.5) ** 2) * (
             np.lib.scimath.sqrt(fltSz ** 2 - (maxxy + 0.5) ** 2)).real
    m2 = (fltSz ** 2 > (maxxy - 0.5) ** 2 + (minxy + 0.5) ** 2) * (minxy + 0.5) + \
         (fltSz ** 2 <= (maxxy - 0.5) ** 2 + (minxy + 0.5) ** 2) * (
             np.lib.scimath.sqrt(fltSz ** 2 - (maxxy - 0.5) ** 2)).real

    sgrid = ((0.5 * (np.arcsin(m2 / fltSz) - np.arcsin(m1 / fltSz)) + 0.25 * (
            np.sin(2 * np.arcsin(m2 / fltSz)) - np.sin(2 * np.arcsin(m1 / fltSz)))) * fltSz ** 2 - \
             (maxxy - 0.5) * (m2 - m1) + (m1 - minxy + 0.5)
             ) * \
            np.logical_or(
                np.logical_and((fltSz ** 2 < (maxxy + 0.5) ** 2 + (minxy + 0.5) ** 2),
                               (fltSz ** 2 > (maxxy - 0.5) ** 2 + (minxy - 0.5) ** 2)),
                np.logical_and(np.logical_and((minxy == 0), (maxxy - 0.5 < fltSz)), (maxxy + 0.5 >= fltSz))
            )

    sgrid = sgrid + ((maxxy + 0.5) ** 2 + (minxy + 0.5) ** 2 < fltSz ** 2)
    sgrid[crad, crad] = min(np.pi * fltSz ** 2, np.pi / 2)

    if ((crad > 0) and (fltSz > crad - 0.5) and (fltSz ** 2 < (crad - 0.5) ** 2 + 0.25)):
        print("yes")
        m1 = (np.lib.scimath.sqrt(fltSz ** 2 - (crad - 0.5) ** 2)).real
        m1n = m1 / fltSz
        sg0 = 2 * ((0.5 * np.arcsin(m1n) + 0.25 * np.sin(2 * np.arcsin(m1n))) * fltSz ** 2 - m1 * (crad - 0.5))
        sgrid[2 * crad, crad] = sg0
        sgrid[crad, 2 * crad] = sg0
        sgrid[crad, 0] = sg0
        sgrid[0, crad] = sg0
        sgrid[2 * crad - 1, crad] = sgrid[2 * crad - 1, crad] - sg0
        sgrid[crad, 2 * crad - 1] = sgrid[crad, 2 * crad - 1] - sg0
        sgrid[crad, 1] = sgrid[crad, 1] - sg0
        sgrid[1, crad] = sgrid[1, crad] - sg0

    sgrid[crad, crad] = min(sgrid[crad, crad], 1)
    h = sgrid / np.sum(sgrid)
    h = (h - np.min(h)) / (np.max(h - np.min(h)))
    return h


def local_gradient_alloc(img_sz: Tuple[int, int], R: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

           PREALLOCATES MATRICES FOR local_gradient
    :param img_sz: tuple; (h, w); h, w - int; height and width of the image for which local gradient is calculated
    :param R: float > 0.5; radius of window filter
    :return: gMatxfft, gMatyfft, sMatfft; three 2d numpy arrays; shape=(h+2*fltSz+1, w+2*fltSz+1);
             2D Fourier transform matrices for calculation of horizontal, vertical and sum of pixels correspondingly
    """
    img_x, img_y = img_sz  # 288, 380
    cR = math.ceil(R - 0.5)
    h = disk_filter(R)
    h = h / np.max(h)

    [g_mat_y, g_mat_x] = np.mgrid[-cR:cR + 1, -cR:cR + 1]
    outsz1, outsz2 = img_x + 2 * cR + 1, img_y + 2 * cR + 1
    gMatxfft = fft2(np.multiply(g_mat_x, h), s=(outsz1, outsz2))
    gMatyfft = fft2(np.multiply(g_mat_y, h), s=(outsz1, outsz2))

    s_mat = np.ones((2 * cR + 1, 2 * cR + 1)) * h
    sMatfft = fft2(s_mat, s=(outsz1, outsz2))
    return gMatxfft, gMatyfft, sMatfft


def local_gradient(
        img: np.ndarray,
        R: float,
        gMatxfft: np.ndarray,
        gMatyfft: np.ndarray,
        sMatfft: np.ndarray,
        thrtype: str,
        thr: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """

           CALCULATES LOCAL GRADIENTS OF THE IMAGE
    :param img: 2d numpy array
    :param R: float > 0.5; radius of window filter
    :param gMatxfft: 2d numpy array
    :param gMatyfft: 2d numpy array
    :param sMatfft: 2d numpy array
           2d Fourier transform matrices for calculation of horizontal, vertical and sum of pixels correspondingly
    :param thrtype: string; 'topfraction' or 'topvalue'; type of threshold to apply
    :param thr: non-negative float; threshold value
    :return: g, gradient, 2d numpy array; magnitude of local gradients
             g_x, gradient in x direction, 2d numpy array; horizontal local gradients
             g_y, gradient in y direction, 2d numpy array; vertial local gradients
             g_thr, threshold gradient, 2d numpy array;
             g_mask, binary threshold gradient, 2d numpy array;
             lsq_data, list of three numpy arrays with shapes: (i, 2), (i, 2), (1, i, 1) where i - non-negative int
    """
    img_x, img_y = img.shape
    cR = math.ceil(R - 0.5)
    outsz = np.array([[2 * cR + 1, img_x], [2 * cR + 1, img_y]])
    img = img + 1  # avoid division by zero

    im_fft = fft2(img, s=(outsz[0, 0] + outsz[0, 1], outsz[1, 0] + outsz[1, 1]))
    # sum of all pixels in the area
    im_sum = np.real(ifft2(np.multiply(im_fft, sMatfft)))
    im_sum = im_sum[outsz[0, 0] - 1: outsz[0, 1], outsz[1, 0] - 1: outsz[1, 1]]

    # x gradient
    g_x = np.real(ifft2(np.multiply(im_fft, gMatxfft)))
    g_x = np.divide(
        g_x[outsz[0, 0] - 1: outsz[0, 1], outsz[1, 0] - 1: outsz[1, 1]],
        im_sum
    )

    # y gradient
    g_y = np.real(ifft2(np.multiply(im_fft, gMatyfft)))
    g_y = np.divide(
        g_y[outsz[0, 0] - 1: outsz[0, 1], outsz[1, 0] - 1: outsz[1, 1]],
        im_sum
    )

    # gradient magnitude
    g = np.sqrt(g_x ** 2 + g_y ** 2)

    if thrtype == 'topfraction':
        cond = np.max(g) / thr
    elif thrtype == 'topvalue':
        cond = thr

    # g_size = g.shape
    mask = g > cond
    g_mask = np.where(mask, 1, 0)
    g_thr = np.multiply(g_mask, g)

    c_r = np.argwhere(mask)
    grad = np.vstack(
        (g_x[mask] + c_r[:, 0],
         g_y[mask] + c_r[:, 1])
    ).T
    v = g[mask]
    lsq_data = [c_r, grad, v.reshape(1, v.shape[0], 1)]
    return g, g_x, g_y, g_thr, g_mask, lsq_data


def lstsqr_lines(p1: np.ndarray, p2: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

               LEAST-SQUARE LINE INTERSECTION
    :param p1: 2d numpy array; shape=(i, 2) where i - non-negative int; first points that define lines
    :param p2: 2d numpy array; shape=(i, 2) where i - non-negative int; second points that define lines
    :param w: 3d numpy array; shape=(1, i, 1) where i - non-negative int; line weights
    :return: c_x_y; numpy array; shape=(2, ); Least-squares solution x, y coordinates of the intersection point
             P; numpy array; shape=(i, 2) where i - non-negative int; coordinates of nearest points on each line
             dR; numpy array; shape=(i,) where i - non-negative int; distance from intersection to each line
    """
    n = p2 - p1
    rows, col = n.shape
    n = n / np.sqrt(np.sum(n ** 2, 1))[:, np.newaxis]
    inn = np.repeat(n.T[:, :, np.newaxis], col, axis=2) * n - np.eye(col)[:, np.newaxis, :]
    inn = inn * w
    r = np.sum(inn, axis=1)
    q = np.matmul(np.vstack((inn[0, :, :], inn[1, :, :])).T, np.hstack((p1[:, 1], p1[:, 0])) + 1)
    c_x_y, _, _, _ = lstsq(r, q, lapack_driver='gelsy')

    # extra outputs
    u = np.sum((np.flip(c_x_y) - (p1 + 1)) * np.fliplr(n), axis=1)
    P = (p1 + 1) + np.repeat(u[:, np.newaxis], col, axis=1) * np.fliplr(n)  # nearest point on each line
    dR = np.sqrt(np.sum((np.flip(c_x_y) - P) ** 2, axis=1))  # distance from intersection to each line
    return c_x_y, P, dR


def z_brt(g_x: np.ndarray, g_y: np.ndarray, c_x_y: np.ndarray) -> float:
    """

           CALCULATE Z COORDINATE OF THE PARTICLE
    :param g_x: gradient in x direction, 2d numpy array; horizontal local gradients
    :param g_y: gradient in y direction, 2d numpy array; vertial local gradients
    :param c_x_y: numpy array; shape=(2, ); dtype=float; least-squares solution x, y coordinates of the intersection point
    :return: z, float; z coordinate of the particle
    """
    try:
        # X horizontal split
        ccol = math.ceil(c_x_y[0] - 0.5)  # find central column
        g_x_left = g_x[:, :ccol - 1]  # split into left and right parts
        g_x_right = g_x[:, ccol:]
        g_x_cSum = np.sum(g_x[:, ccol - 1])  # find sum of central column and corresponding fractions of a pixel
        frxL = c_x_y[0] - (ccol - 0.5)
        frxR = 1 - frxL
        g_x_lSum = np.sum(g_x_left) + g_x_cSum * frxL
        g_x_rSum = np.sum(g_x_right) + g_x_cSum * frxR
        zVx = g_x_rSum - g_x_lSum  # calculate z value from x gradient

        # Y vertical split
        crow = math.ceil(c_x_y[1] - 0.5)  # find central row
        g_y_top = g_y[:crow - 1, :]  # split into top and bottom parts
        g_y_bottom = g_y[crow:, :]
        g_y_cSum = np.sum(g_y[crow - 1, :])  # find sum of central row and corresponding fractions of a pixel
        fryT = c_x_y[1] - (crow - 0.5)
        fryB = 1 - fryT
        g_y_tSum = np.sum(g_y_top) + g_y_cSum * fryT
        g_y_bSum = np.sum(g_y_bottom) + g_y_cSum * fryB
        zVy = g_y_bSum - g_y_tSum  # calculate z value from y gradient
        return (zVx + zVy) / 2
    except IndexError:
        print("to get z coordinate value, please choose smaller value for R")


def z_ast(g_thr: np.ndarray, g_x: np.ndarray, g_y: np.ndarray, c_x_y: np.ndarray, positiveAngle: int) -> float:
    """

           CALCULATE Z COORDINATE OF THE FLUORESCENT PARTICLE
    :param g_thr: threshold gradient, 2d numpy array;
    :param g_x: gradient in x direction, 2d numpy array; horizontal local gradients
    :param g_y: gradient in y direction, 2d numpy array; vertial local gradients
    :param c_x_y: numpy array; shape=(2, ); Least-squares solution x, y coordinates of the intersection point
    :param positiveAngle: int; angle of the major axes (in degrees) of the spot
                          It is used to discriminate positive and negative displacement
                          of the particle from the in-focus position define positive
                          (should be measured from the positive direction of x-axis)

    :return: zV; z-value, float;

    AxMJR, AxMNR - major and minor axes (distances between centers of
                      opposite halves of the image - TOP vs BOTTOM, LEFT vs RIGHT)

    The image of local gradients (g_thr) is splitted into 4 parts: TOP, LEFT, BOTTOM, RIGHT, relative to the particle
    center (cx,cy). The least-square intersection of all gradient lines (g_x, g_y) is calculated for each part. These
    centers are forming two lines: TOP-BOTTOM and LEFT-RIGHT which are called axes (similar to ellipses axes). The
    major of two axes represents the magnitude of the z-value. The sign of the z-value is determined according to
    the positiveAngle value which should correspond to the position of the long axis of the elliptical image of the
    particle. positiveAngle is not required to be accurate, however, setting it to ~45 degrees off the real angle will
    create sporadic change in the sign and, therefore, in z-value.
    """

    g_size = g_thr.shape

    # X horizontal split
    ccol = math.ceil(c_x_y[0] - 0.5)  # find central column
    # find fractions of a pixel for central column
    frxL = c_x_y[0] - (ccol - 0.5)
    frxR = 1 - frxL
    # create horizontally splitted parts of the local gradient image
    g_thr_right = g_thr * np.pad(np.ones((g_size[0], g_size[1] - ccol + 1)), [(0, 0), (ccol - 1, 0)], mode='constant')
    g_thr_right[:, ccol - 1] = g_thr_right[:, ccol - 1] * frxR
    g_thr_left = g_thr * np.pad(np.ones((g_size[0], ccol)), [(0, 0), (0, g_size[1] - ccol)], mode='constant')
    g_thr_left[:, ccol - 1] = g_thr_left[:, ccol - 1] * frxL

    # Y vertical split
    crow = math.ceil(c_x_y[1] - 0.5)  # find central row
    frxT = c_x_y[1] - (crow - 0.5)  # find fractions of a pixel for central row
    frxB = 1 - frxT
    # create vertically splitted parts of the local gradient image
    g_thr_bottom = g_thr * np.pad(np.ones((g_size[0] - crow + 1, g_size[1])), [(crow - 1, 0), (0, 0)], mode='constant')
    g_thr_bottom[crow - 1, :] = g_thr_bottom[crow - 1, :] * frxB
    g_thr_top = g_thr * np.pad(np.ones((crow, g_size[1])), [(0, g_size[0] - crow), (0, 0)], mode='constant')
    g_thr_top[crow - 1, :] = g_thr_top[crow - 1, :] * frxT
    # calculate least-square fit of gradient lines for each part
    g_z = (g_thr_top, g_thr_left, g_thr_bottom, g_thr_right)
    h_x = np.zeros(4)
    h_y = np.zeros(4)

    # loop through each half
    for j in range(4):
        mask = g_z[j] > 0
        c_r = np.argwhere(mask)
        grad = np.vstack(
            (g_x[mask] + c_r[:, 0],
             g_y[mask] + c_r[:, 1])
        ).T
        v = g_z[j][mask]
        lsq_data = [c_r, grad, v.reshape(1, v.shape[0], 1)]
        c_x_y_Local, _, _ = lstsqr_lines(lsq_data[0], lsq_data[1], lsq_data[2])
        cxLocal, cyLocal = c_x_y_Local

        h_x[j] = cxLocal - c_x_y[0]
        h_y[j] = cyLocal - c_x_y[1]

    ax1d = np.sqrt((h_x[0] - h_x[2]) ** 2 + (h_y[0] - h_y[2]) ** 2)  # 1st axis length
    ax2d = np.sqrt((h_x[1] - h_x[3]) ** 2 + (h_y[1] - h_y[3]) ** 2)  # 2nd axis length

    # find major and minor axes length
    AxMJR = max(ax1d, ax2d)
    AxMNR = min(ax1d, ax2d)

    p1 = np.rad2deg(np.sin(2 * positiveAngle)) * (h_x[0] - h_x[2]) + np.rad2deg(np.cos(2 * positiveAngle)) * (
            h_y[0] - h_y[2])
    p2 = np.rad2deg(np.sin(2 * (positiveAngle + 90))) * (h_y[3] - h_y[1]) - np.rad2deg(
        np.cos(2 * (positiveAngle + 90))) * (h_x[3] - h_x[1])
    focus_sign = np.sign(p1 + p2)
    zV = focus_sign * AxMJR
    return zV


def z_dh(pnts: np.ndarray, mid_rng: int) -> float:
    """

           CALCULATE Z COORDINATE OF THE FLUORESCENT PARTICLE IN DOUBLE-HELIX-BASED MICROSCOPY
    :param pnts: 2d numpy array; coordinates of points in the image (2-column vector)
    :param mid_rng: int; indication of mid-range angle of rotation [-180..180]

    :return: theta; z-value, float;

    Finds the orientation of the image using moments of the image
    """

    x = pnts[:, 1]
    y = -pnts[:, 0]

    # Calculate moments
    M00 = np.size(x)
    M10 = np.sum(x)
    M01 = np.sum(y)
    M11 = np.sum(x * y)
    M20 = np.sum(x ** 2)
    M02 = np.sum(y ** 2)

    # Find center
    cx = M10 / M00
    cy = M01 / M00

    # Calculate central moments
    u11 = M11 / M00 - cx * cy
    u20 = M20 / M00 - cx ** 2
    u02 = M02 / M00 - cy ** 2

    # Calculate the angle
    theta = 0.5 * np.arctan2(2 * u11, (u20 - u02)) * 180 / np.pi

    # Correct angle to the specified range
    fromRng = mid_rng - 90
    toRng = mid_rng + 90
    if theta < fromRng:
        theta = theta + 180
    elif theta > toRng:
        theta = theta - 180
    return theta


def get_position_brightfield(
        img_arr: np.ndarray,
        thrtype: str,
        thr: float,
        R: float,
        G: tuple = None,
        z_pos: bool = True,
        draw: bool = False
) -> Tuple[float, float, float]:
    """
    :param img_arr: np.ndarray; 2d image array
    :param thrtype: string; 'topfraction' or 'topvalue'
    :param thr: non-negative float; threshold level
    :param R: float > 0.5; radius of window filter
    :param G: None by default or tuple with 3 already precalculated 2d numpy.ndarrays - 2D fourier transform matrices for
              calculation of horizontal, vertical gradients and sum of pixels correspondingly
    :param z_pos: bool; True by default; whether to calculate z position
    :param draw: bool; False by default; whether to show a drawing
    :return: tuple with 3 floats; x,y,z coordinate particle center
    """
    if G:
        gMatxfft, gMatyfft, sMatfft = G
    else:
        gMatxfft, gMatyfft, sMatfft = local_gradient_alloc(img_sz=img_arr.shape, R=abs(R))

    g, g_x, g_y, g_thr, g_mask, lsq_data = local_gradient(img_arr, abs(R), gMatxfft, gMatyfft, sMatfft, thrtype, thr)
    c_x_y, P, dR = lstsqr_lines(lsq_data[0], lsq_data[1], lsq_data[2])

    if draw:
        fig = plt.figure(figsize=(20, 10))
        for idx, item in enumerate([img_arr, g, g_x, g_y, g_thr, g_mask, 0, img_arr]):
            fig.add_subplot(2, 4, 1 + idx)
            if idx == 6:
                p1 = lsq_data[0] + 1
                p2 = lsq_data[1] + 1
                plt.scatter(P[:, 1], P[:, 0], color='red', s=0.5)
                for i in range(p1.shape[0]):
                    plt.plot([p1[i, 1], p2[i, 1]], [p1[i, 0], p2[i, 0]], 'ro-', linewidth=0.5, markersize=0)
            elif idx == 7:
                fig.add_subplot(2, 4, 1 + idx)
                plt.imshow(item, cmap='gray', zorder=1)
                plt.scatter(c_x_y[0] + abs(R) - 1, c_x_y[1] + abs(R) - 1, color='red', marker="x", zorder=2, s=100)
            else:
                plt.imshow(item, cmap='gray')
        fig.tight_layout()
        plt.show()

    if z_pos:
        z = z_brt(g_x, g_y, c_x_y)
        return c_x_y[0] + abs(R), c_x_y[1] + abs(R), z
    return c_x_y[0] + abs(R), c_x_y[1] + abs(R), 0.0


def get_position_astigmatism(
        img_arr: np.ndarray,
        thrtype: str,
        thr: float,
        R: float,
        positiveAngle: int,
        G: tuple = None,
        roi: list = None,
        z_pos: bool = True
) -> Tuple[float, float, float]:
    """
    :param img_arr: np.ndarray; 2d image array
    :param thrtype: string; 'topfraction' or 'topvalue'
    :param thr: non-negative float; threshold level
    :param R: float > 0.5; radius of window filter
    :param positiveAngle: int; angle in degrees of positive direction of the particle's image (measured from
                          positive x axis in counter-clockwise direction)
    :param G: None by default or tuple with 3 already precalculated 2d numpy.ndarrays - 2D fourier transform matrices for
              calculation of horizontal, vertical gradients and sum of pixels correspondingly
    :param roi: None by default or list with ints of length 4; region of interest to select an individual fluorophore
                from the image, should be greater than zero and less than corresponding image size
    :param z_pos: bool; whether to calculate z position; True by default
    :return: x, y, z: tuple with 3 floats; x, y, z coordinates of fluorescent particles
    """
    if roi == None:
        roi[0], roi[2] = 1, 1
        roi[1], roi[3] = img_arr.shape[0], img_arr.shape[1]
    if G:
        gMatxfft, gMatyfft, sMatfft = G
    else:
        # Precalculate matrices for local gradient calculations
        gMatxfft, gMatyfft, sMatfft = local_gradient_alloc(img_sz=(roi[1] - roi[0] + 1, roi[3] - roi[2] + 1), R=abs(R))

    im_analyze = img_arr[roi[0] - 1: roi[1], roi[2] - 1: roi[3]]  # apply region of interest
    # calculate local gradients images
    g, g_x, g_y, g_thr, g_mask, lsq_data = local_gradient(im_analyze, abs(R), gMatxfft, gMatyfft, sMatfft, thrtype, thr)
    # find center of symmetry of the particle
    c_x_y, P, dR = lstsqr_lines(lsq_data[0], lsq_data[1], lsq_data[2])
    # correct determined positiones for the reduction in the image size
    cR = math.ceil(R - 0.5)
    if z_pos:
        # calculate z-value
        z = z_ast(g_thr, g_x, g_y, c_x_y, positiveAngle)
        return c_x_y[0] + cR, c_x_y[1] + cR, z

    return c_x_y[0] + cR, c_x_y[1] + cR, 0.0


def get_position_doublehelix(
        img_arr: np.ndarray,
        thrtype: str,
        thr: float,
        R: float,
        mid_rng: int,
        G: tuple = None,
        roi: list = None,
        z_pos: bool = True
) -> Tuple[float, float, float]:
    """
    :param img_arr: np.ndarray; 2d image array
    :param thrtype: string; 'topfraction' or 'topvalue'
    :param thr: non-negative float; threshold level
    :param R: float > 0.5; radius of window filter
    :param mid_rng: int; indication of mid-range angle of rotation [-180..180] (measured from positive x axis in
           counter-clockwise direction)
    :param G: None by default or tuple with 3 already precalculated 2d numpy.ndarrays - 2D fourier transform matrices for
              calculation of horizontal, vertical gradients and sum of pixels correspondingly
    :param roi: None by default or list with ints of length 4; region of interest to select an individual fluorophore
                from the image, should be greater than zero and less than corresponding image size
    :param z_pos: bool; whether to calculate z position; True by default
    :return: x, y, z: tuple with 3 floats; x, y, z coordinates of fluorescent particles
    """
    if roi == None:
        roi[0], roi[2] = 1, 1
        roi[1], roi[3] = img_arr.shape[0], img_arr.shape[1]
    if G:
        gMatxfft, gMatyfft, sMatfft = G
    else:
        # Precalculate matrices for local gradient calculations
        gMatxfft, gMatyfft, sMatfft = local_gradient_alloc(img_sz=(roi[1] - roi[0] + 1, roi[3] - roi[2] + 1), R=abs(R))

    im_analyze = img_arr[roi[0] - 1: roi[1], roi[2] - 1: roi[3]]  # apply region of interest
    # calculate local gradients images
    g, g_x, g_y, g_thr, g_mask, lsq_data = local_gradient(im_analyze, abs(R), gMatxfft, gMatyfft, sMatfft, thrtype, thr)
    # find center of symmetry of the particle
    c_x_y, P, dR = lstsqr_lines(lsq_data[0], lsq_data[1], lsq_data[2])
    # correct determined positiones for the reduction in the image size
    cR = math.ceil(R - 0.5)
    if z_pos:
        # calculate z-value
        z = z_dh(lsq_data[0], mid_rng)
        return c_x_y[0] + cR, c_x_y[1] + cR, z

    return c_x_y[0] + cR, c_x_y[1] + cR, 0.0
