"""
Here are implemented all the main functions of the local gradient method

Authors: Oleksandr Perederii, Anatolii Kashchuk
2022
"""

import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.fft import fft2, ifft2
from scipy.linalg import lstsq
import numpy as np
import math

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

            CALCULATE X, Y, Z COORDINATEs OF THE PARTICLE IN BRIGHTFIELD MICROSCOPY
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
        G: tuple = None,
        roi: list = None,
        z_pos: bool = True,
        positiveAngle: int = 90
) -> Tuple[float, float, float]:
    """

            CALCULATE X, Y, Z COORDINATEs OF THE FLUORESCENT PARTICLE IN FLURESCENT MICROSCOPE
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
        G: tuple = None,
        roi: list = None,
        z_pos: bool = True,
        mid_rng: int = 91
) -> Tuple[float, float, float]:
    """

            CALCULATE X, Y, Z COORDINATEs OF THE FLUORESCENT PARTICLE IN DOUBLE-HELIX-BASED MICROSCOPY
    :param img_arr: np.ndarray; 2d image array
    :param thrtype: string; 'topfraction' or 'topvalue'
    :param thr: non-negative float; threshold level
    :param R: float > 0.5; radius of window filter
    :param G: None by default or tuple with 3 already precalculated 2d numpy.ndarrays - 2D fourier transform matrices for
              calculation of horizontal, vertical gradients and sum of pixels correspondingly
    :param roi: None by default or list with ints of length 4; region of interest to select an individual fluorophore
                from the image, should be greater than zero and less than corresponding image size
    :param z_pos: bool; whether to calculate z position; True by default
    :param mid_rng: int; indication of mid-range angle of rotation [-180..180] (measured from positive x axis in
           counter-clockwise direction)
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


def local_gradient_multi(
        img: np.ndarray,
        R: float,
        epsilon: float,
        minpts: int,
        thrtype: str,
        thr: float,
        G: tuple = None
) -> np.ndarray:
    """

           CALCULATES z-value FOR A FLUORESCENT PARTICLE IN ASTIGMATISM-BASED MICROSCOPY
    :param img: 2d numpy array
    :param R: float > 0.5; radius of window filter
    :param epsilon: float > 0; neighborhood search radius: The maximum distance between two samples for one to be
                    considered as in the neighborhood of the other (DBSCAN)
    :param minpts: int > 0; minimum number of neighbors minpts required to identify a core point
    :param thrtype: string; 'topfraction' or 'topvalue'; type of threshold to apply
    :param thr: non-negative float; threshold value
    :param G: None by default or tuple with 3 already precalculated 2d numpy.ndarrays - 2D fourier transform matrices for
              calculation of horizontal, vertical gradients and sum of pixels correspondingly
    :return: coord; numpy array; shape=(number_of_clusters, 2); array contains the coordinates of all particles in the
                    image

    """
    if G:
        gMatxfft, gMatyfft, sMatfft = G
    else:
        gMatxfft, gMatyfft, sMatfft = local_gradient_alloc(img_sz=img.shape, R=abs(R))

    # calculate local gradients images
    g, _, _, _, g_mask, lsq_data = local_gradient(img, abs(R), gMatxfft, gMatyfft, sMatfft, thrtype, thr)

    # cluster data
    x_y_coord = lsq_data[0]
    idx = DBSCAN(eps=epsilon, min_samples=minpts).fit_predict(x_y_coord)                     # get labels for each point
    n_clusters_ = max(idx) + 1                                                 # len(set(idx)) - (1 if -1 in idx else 0)
    coord = np.zeros((n_clusters_, 2))

    # go through all detected clusters
    for i in range(n_clusters_):
        c_x_y, _, _ = lstsqr_lines(lsq_data[0][idx == i, :], lsq_data[1][idx == i, :], lsq_data[2][:, idx == i, :])
        coord[i, :] = c_x_y
    cR = math.ceil(R - 0.5)

    idx = np.argsort(coord[:, 0])                                                             # sort by first coordinate
    coord = coord[idx, :]
    return coord + cR


def detect_trj(
        c_arr: np.ndarray,
        dc: int,
        dfr: int,
        Nfr_min: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """

            LINKS POSITION DATA INTO TRAJECTORIES

    :param c_arr: 3d numpy array; shape=(N=index_of_frame, n=index_of_particle, 2); N frames, each containing x-y
                  coordinates of n particles
    :param dc: int > 0; maximum distance from the detected particle to look for linked particles in other frames
    :param dfr: int > 0; number of frames to look for linked particles
    :param Nfr_min: int > 0; minimum number of frames in trajectory
    :return:
            t_trj_ids:
            t_frames:
            t_xy: 3d numpy array; shape=(N=index_of_frame, n=index_of_particle, 2=xy coordinate);
            t_trj_num:
    """
    num_of_frames = c_arr.shape[0]
    num_of_particles = c_arr.shape[1]
    trj_num = c_arr.shape[1]

    t_ids = np.empty((num_of_frames, num_of_particles), dtype=int)
    t_frames = np.empty((num_of_frames, num_of_particles), dtype=int)

    t_ids[0, :] = np.arange(num_of_particles)
    t_frames[0, :] = np.zeros((num_of_particles))

    for frame in range(1, num_of_frames):
        for particle in range(num_of_particles):
            for k in range(1, dfr + 1):
                if frame + 1 - k <= 0:
                    trj_num += 1
                    t_ids[frame, particle] = trj_num - 1
                    break
                # find minimum distance and index of the closest particle
                arr = np.sqrt(np.sum((c_arr[frame, particle, :] - c_arr[frame - k, :, :]) ** 2, axis=1))
                m = np.min(arr)
                ind = np.argmin(arr)

                if m <= dc:
                    # if within a specified distance - mark current particle as a part of trajectory
                    mtx_temp = t_ids[frame - k, ind]
                    t_ids[frame, particle] = mtx_temp
                    break
                else:
                    # if all frames checked and no particles is close enough, create new trajectory id
                    if k == dfr:
                        trj_num += 1
                        t_ids[frame, particle] = trj_num - 1
        t_frames[frame, :] = frame * np.ones(particle + 1)

    # trajectory ids for all detected points
    trj_ids = np.sort(np.concatenate(t_ids, axis=0), axis=0)                      # (num_of_frames * num_of_particles, )

    # number of frames for each trajectory
    n_frames = np.diff(
        (np.diff(
            np.concatenate((np.array([-1]), trj_ids, np.array([trj_num])), axis=0),
            axis=0
        ) != 0).nonzero()[0]
    )

    # filter trajectories by frame number
    trj_filt = (n_frames >= Nfr_min).nonzero()[0]
    num_of_trj = trj_filt.shape[0]

    t_ext_ids = np.concatenate(t_ids, axis=0)                                     # (num_of_frames * num_of_particles, )
    t_ext_frames = np.concatenate(t_frames, axis=0)                               # (num_of_frames * num_of_particles, )
    t_ext_xy = np.concatenate(c_arr, axis=0)                                     # (num_of_frames * num_of_particles, 2)

    t_trj_ids, t_frames, t_xy = None, None, None
    t_trj_num = np.empty((num_of_trj))

    for p, t in enumerate(trj_filt):
        mask = t_ext_ids == t
        if p == 0:
            t_trj_ids = t_ext_ids[mask]
            t_frames = t_ext_frames[mask]
            t_xy = t_ext_xy[mask][..., np.newaxis]
        else:
            t_trj_ids = np.vstack((t_trj_ids, t_ext_ids[mask]))
            t_frames = np.vstack((t_frames, t_ext_frames[mask]))
            t_xy = np.concatenate((t_xy, t_ext_xy[mask][..., np.newaxis]), axis=2)
        t_trj_num[p] = t
    return t_trj_ids, t_frames, t_xy, t_trj_num


def _precalc_minV_indV(
        minV: np.ndarray,
        indV: np.ndarray,
        c_arr: np.ndarray,
        k: int,
        dc: int,
        dfr: int,
        nonzero_ids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """

            PRECALCULATION OF minV AND indV MATRICES

    :param minV: 3d np.ndarray of floats; shape:(number of frames, number of particles, dfr)
    :param indV: 3d np.ndarray of ints; shape:(number of frames, number of particles, dfr)
    :param c_arr: 3d np.ndarray of floats; shape: (num_of_frames, num_of_particles, 2)
    :param k: int > 0; integer in range(1, dfr + 1)
    :param dc: int > 0; maximum distance from the detected particle to look for linked particles in other frames
    :param dfr: int > 0; number of frames to look for linked particles
    :param nonzero_ids: 2d np.ndarray of bool values; shape:(number of frames, number of particles)
                 True value indicates that for a specific (i, j) minV[i, j, :k] array contains
                 all zeros. False - array contains a nonzero value c equal to -1 or > 0
    :return:
            minV, indV: np.ndarray of floats, np.ndarray of ints; shape:(number of frames, number of particles, dfr)
    """
    # if sum of abs values of minV array along last dimension gives nonzero matrix
    if np.all(nonzero_ids == False) or k > dfr:
        return minV, indV
    m1 = c_arr[:, :, np.newaxis, :]
    m2 = c_arr[:, np.newaxis, :, :]

    diff = np.subtract(
        m1[k:, ...],
        m2[:-k, ...],
        where=np.repeat(nonzero_ids[:, :, np.newaxis], 2, axis=2)[:, :, np.newaxis, :]
    )                                                # get difference only for minV[..., :k] elements that ara all zeros

    diff_sum = np.sqrt(np.sum(diff ** 2, axis=3))
    m = np.min(diff_sum, axis=2)
    e = m <= dc                                                              # fill minV and indV arrays only if m <= dc
    ij = np.argwhere(np.logical_and(nonzero_ids == True, e))                                   # and nonzero_ids == True
    minV[ij[:, 0] + k, ij[:, 1], k - 1] = m[ij[:, 0], ij[:, 1]]
    indV[ij[:, 0] + k, ij[:, 1], k - 1] = np.argmin(diff_sum, axis=2)[ij[:, 0], ij[:, 1]]
    # recalculate nonzero_ids for function recursive call with incremented k
    new_nonzero_ids = np.sum(abs(minV[k + 1:, :, :k]), axis=2) == 0
    return _precalc_minV_indV(minV, indV, c_arr, k + 1, dc, dfr, new_nonzero_ids)


def _first_nonzero(
        arr: np.ndarray,
        axis: int,
        invalid_val: int = -1
) -> np.ndarray:
    """

          FIND INDICES OF FIRST NON-ZERO ELEMENT ALONG AXIS

    arr: np.ndarray
    axis: int
    invalid_val: int, float; the value that will contain the result matrix if all the elements along axis are zeros

    return:
            np.ndarray with indices of first nonzero element of array arr along axis
    """
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def fast_detect_trj(
        c_arr: np.ndarray,
        dc: int,
        dfr: int,
        Nfr_min: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """

            faster implementation of detect_trj function
            LINKS POSITION DATA INTO TRAJECTORIES

    :param c_arr: 3d numpy array; shape=(N=index_of_frame, n=index_of_particle, 2); N frames, each containing x-y
                  coordinates of n particles
    :param dc: int > 0; maximum distance from the detected particle to look for linked particles in other frames
    :param dfr: int > 0; number of frames to look for linked particles
    :param Nfr_min: int > 0; minimum number of frames in trajectory
    :return:
            t_trj_ids:
            t_frames:
            t_xy: 3d numpy array; shape=(N=index_of_frame, n=index_of_particle, 2=xy coordinate);
            t_trj_num:
    """
    num_of_frames = c_arr.shape[0]
    num_of_particles = c_arr.shape[1]
    trj_num = c_arr.shape[1]

    t_ids = np.zeros((num_of_frames, num_of_particles), dtype=int)
    t_frames = np.zeros((num_of_frames, num_of_particles), dtype=int)

    t_ids[0, :] = np.arange(num_of_particles)
    t_frames[0, :] = np.zeros((num_of_particles))

    minV = np.zeros((num_of_frames, num_of_particles, dfr))
    indV = np.zeros((num_of_frames, num_of_particles, dfr), dtype=int)

    r, c, h = np.indices((num_of_frames, num_of_particles, dfr))
    minV[r == h] = -1  # frame + 1 - k <= 0; k in range(1, dfr)

    start_ids = minV[:, :, 0] == 0
    """ c_arr shape: (num_of_frames, num_of_particles, 2) """
    """ minV, indV shape: (num_of_frames, num_of_particles, dfr) """
    minV, indV = _precalc_minV_indV(minV, indV, c_arr, 1, dc, dfr, start_ids[1:, :])

    # get matrice of indexes of the first non-zero elements of minV array along 2 axis and matrice of its values
    frst_nonzero_id = _first_nonzero(minV, axis=2, invalid_val=-1)
    values = np.squeeze(np.take_along_axis(minV, frst_nonzero_id[:, :, np.newaxis], 2))

    t_ids = np.zeros((num_of_frames, num_of_particles), dtype=int)
    t_ids[0, :] = np.arange(num_of_particles)
    # set values for cases when frame + 1 - k <= 0 and k == dfr
    t_ids[1:, :][np.where(np.logical_or(values[1:, :] == -1, values[1:, :] == 0))] = \
        np.arange(num_of_particles, num_of_particles + (values[1:, :] == -1).sum() + (values[1:, :] == 0).sum())

    t_frames[1:, :] = np.arange(1, num_of_frames)[:, np.newaxis] * np.ones(num_of_particles)

    # set values for the last case
    for fr_part in list(zip(*np.where(t_ids == 0))):
        frame, particle = fr_part
        m = minV[frame, particle, :]
        k = frst_nonzero_id[frame, particle]
        ind = indV[frame, particle, k]
        mtx_temp = t_ids[frame - k - 1, ind]
        t_ids[frame, particle] = mtx_temp

    # trajectory ids for all detected points
    trj_ids = np.sort(np.concatenate(t_ids, axis=0), axis=0)                      # (num_of_frames * num_of_particles, )

    # number of frames for each trajectory
    n_frames = np.diff(
        (np.diff(
            np.concatenate((np.array([-1]), trj_ids, np.array([trj_num])), axis=0),
            axis=0
        ) != 0).nonzero()[0]
    )

    # filter trajectories by frame number
    trj_filt = (n_frames >= Nfr_min).nonzero()[0]
    num_of_trj = trj_filt.shape[0]

    t_ext_ids = np.concatenate(t_ids, axis=0)                                     # (num_of_frames * num_of_particles, )
    t_ext_frames = np.concatenate(t_frames, axis=0)                               # (num_of_frames * num_of_particles, )
    t_ext_xy = np.concatenate(c_arr, axis=0)                                     # (num_of_frames * num_of_particles, 2)

    t_trj_ids, t_frames, t_xy = None, None, None
    t_trj_num = np.empty((num_of_trj))

    for p, t in enumerate(trj_filt):
        mask = t_ext_ids == t
        if p == 0:
            t_trj_ids = t_ext_ids[mask]
            t_frames = t_ext_frames[mask]
            t_xy = t_ext_xy[mask][..., np.newaxis]
        else:
            t_trj_ids = np.vstack((t_trj_ids, t_ext_ids[mask]))
            t_frames = np.vstack((t_frames, t_ext_frames[mask]))
            t_xy = np.concatenate((t_xy, t_ext_xy[mask][..., np.newaxis]), axis=2)
        t_trj_num[p] = t
    return t_trj_ids, t_frames, t_xy, t_trj_num