"""
Base class ParticleDetector
Provides methods for calculation particle coordinates in fluorescent and brightfield microscopes.

Authors: Oleksandr Perederii, Anatolii Kashchuk
"""

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
#import os
import PIL
import time
from functools import wraps
from typing import Tuple

from local_gradient_math import local_gradient_alloc, local_gradient, lstsqr_lines, z_value, z_fluor, \
                                get_position_brightfield, get_position_fluorescent
from validators import FilenameValidator, ThrtypeValidator, ThrValidator, RValidator, RoiValidator, PosAngleValidator, \
                       DzValidator, ZValidator, validate


class ParticleDetector:

    @classmethod
    @validate
    def get_pos_fluorescent(
            cls,
            filename: str,
            thrtype: str,
            thr: float,
            R: float,
            positiveAngle: int,
            roi: list = [0, 0, 0, 0],
            dz: float = 0.02,
            z0: int = -1,
            z_pos: bool = True,
            draw: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        :param filename: string; path to the image file, may be multi-image
        :param thrtype: string; 'topfraction' or 'topvalue'
        :param thr: non-negative float; threshold level
        :param R: float > 0.5; radius of window filter
        :param roi: list with ints of length 4; region of interest to select an individual fluorophore from the image,
                                                should be greater than zero and less than corresponding image size
        :param positiveAngle: int; angle in degrees of positive direction of the particle's image (measured from
                                positive x axis in counter-clockwise direction)
        :param dz: float; z step between images
        :param z0: int; first image position
        :param draw: bool; whether to show a drawing
        :param z_pos: bool; whether to calculate z position
        :return: x, y, zV, t: 1d numpy arrays; x, y, z coordinates of fluorescent particles and execution time
        """
        img = PIL.Image.open(filename)

        roi = list(map(int, roi))
        fst_img = PIL.ImageSequence.Iterator(img)[0]
        fst_dim, scd_dim = np.array(fst_img).shape[0], np.array(fst_img).shape[1]
        # if roi size are incorrect - set them to full image size
        if (roi == [0, 0, 0, 0]) or (0 < roi[1] < roi[0] < fst_dim) or (0 < roi[3] < roi[2] < scd_dim):
            roi[0], roi[2] = 1, 1
            roi[1], roi[3] = fst_dim, scd_dim
        # Precalculate matrices for local gradient calculations
        gMatxfft, gMatyfft, sMatfft = local_gradient_alloc(img_sz=(roi[1] - roi[0] + 1, roi[3] - roi[2] + 1), R=abs(R))

        x, y, zV, t = np.array([]), np.array([]), np.array([]), np.array([])
        for i, page in enumerate(PIL.ImageSequence.Iterator(img)):
            start = time.time()
            im = np.array(page)
            if im.ndim == 3:
                im = im[:, :, 0]

            x_i, y_i, z_i = get_position_fluorescent(im, thrtype, thr, R, positiveAngle,
                                                     (gMatxfft, gMatyfft, sMatfft), roi, z_pos)
            end = time.time()
            x = np.append(x, x_i)
            y = np.append(y, y_i)
            zV = np.append(zV, z_i)
            t = np.append(t, end - start)

        if draw:
            # apply polynomial fit to z - value
            xplot = np.arange(z0, z0 + dz * zV.shape[0], dz)
            p_coef, residuals, rank, singular_vals, rcond = np.polyfit(x=xplot, y=zV, deg=4, full=True)
            error = np.polyval(p_coef, xplot)

            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=("X position", "Y position", "Z value", f"Execution time, t_av={np.average(t)}")
            )
            for idx, item in enumerate(((x, "z-position, mkm", "x, pxls", "x position, pxls VS z-position, mkm "),
                                        (y, "z-position, mkm", "y, pxls", "y position, pxls VS z-position, mkm "),
                                        (zV, "z-position, mkm", "z-value", "z-value VS z-position, mkm"),
                                        (t, "z-position, mkm", "t, seconds", "t, seconds VS z-position, mkm"))):
                if idx == 2:
                    fig.add_trace(
                        go.Scatter(x=xplot, y=error, name='error', line=dict(width=1)),  # , mode='lines+markers'
                        row=idx // 2 + 1, col=idx % 2 + 1
                    )
                    fig.add_trace(
                        go.Scatter(x=xplot, y=item[0], mode="markers", marker=dict(size=4), name=item[3]),
                        row=idx // 2 + 1, col=idx % 2 + 1
                    )
                else:
                    fig.add_trace(
                        go.Scatter(x=xplot, y=item[0], line=dict(width=1), name=item[3]),
                        row=idx // 2 + 1, col=idx % 2 + 1
                    )

                fig['layout'][f'xaxis{idx + 1}']['title'] = item[1]
                fig['layout'][f'yaxis{idx + 1}']['title'] = item[2]

            fig.update_layout(height=600, width=1400, title_text="Fluorescent xyz")
            fig.show()

        return x, y, zV, t


    @classmethod
    @validate
    def get_pos_brightfield(
            cls,
            filename: str,
            thrtype: str,
            thr: float,
            R: float,
            z_pos: bool = True,
            draw: bool = False
    ) -> Tuple[float, float, float]:
        """
        :param filename: string; path to the image file
        :param thrtype: string; 'topfraction' or 'topvalue'
        :param thr: non-negative float; threshold level
        :param R: float > 0.5; radius of window filter
        :param z_pos: bool; whether to calculate z position
        :param draw: bool; whether to show a drawing
        :return: tuple with 3 floats; x,y,z coordinate particle center
        """
        img = cv2.imread(filename)
        img_arr = img[:, :, 0]
        x, y, z = get_position_brightfield(img_arr, thrtype, thr, R, z_pos=z_pos, draw=draw)
        return x, y, z



