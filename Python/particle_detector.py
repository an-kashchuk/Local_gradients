"""Base class ParticleDetector Provides methods for calculation particle coordinates in astigmatism and double-helix
fluorescent microscopy and in brightfield microscopes.

Authors: Oleksandr Perederii, Anatolii Kashchuk
2022
"""

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import cv2
# import os
import PIL
import time
from typing import Callable, Tuple

from local_gradient_math import local_gradient_alloc, get_position_brightfield, get_position_astigmatism, \
     get_position_doublehelix, local_gradient_multi, detect_trj
from validators import validate

from multiprocessing import Pool, cpu_count
from functools import partial



class ParticleDetector:

    @staticmethod
    def _parallel(func):
        """

                WRAPS A FUNCTION FOR PARALLEL COMPUTATION
        :param func: callable; arguments to the function should be passed as a sequence of lists containing arguments;
                     [[arg_0_0, arg_0_1, ...], [arg_1_0, arg_1_1, ...], ...]
        :return: wrapper function
        """
        def wrapper(iterable):
            with Pool(cpu_count()) as pool:
                result = np.array(
                    pool.map(func, iterable)
                )
            return result
        return wrapper

    @staticmethod
    def _multiframeimg_to_arr(filename):
        """

                CONVERT MULTIFRAME IMAGE INTO LIST OF ARRAYS
        :param filename: string; path to the image file, may be multi-image
        :return: list that contains image np.ndarray-s
        """
        img = PIL.Image.open(filename)
        img_lst = []
        for i, page in enumerate(PIL.ImageSequence.Iterator(img)):
            img_lst.append(np.array(page))
        return img_lst

    @staticmethod
    def _normalize_roi_for_image(roi, img):
        """

                PREPROCESS roi ARGUMENT
        :param roi: list with ints of length 4; region of interest to select an individual fluorophore from the image,
                    should be greater than zero and less than corresponding image size
        :param img: an image object <<class PIL.Image.Image>>
        :return: roi: list with ints of length 4 that corresponds to the selected area of the image or coordinates
                      of the entire image otherwise
        """
        roi = list(map(int, roi))
        fst_img = PIL.ImageSequence.Iterator(img)[0]
        fst_dim, scd_dim = np.array(fst_img).shape[0], np.array(fst_img).shape[1]
        # if roi size are incorrect - set them to full image size
        if (roi == [0, 0, 0, 0]) or (0 < roi[1] < roi[0] < fst_dim) or (0 < roi[3] < roi[2] < scd_dim):
            roi[0], roi[2] = 1, 1
            roi[1], roi[3] = fst_dim, scd_dim
        return roi

    @staticmethod
    def _process_image(img, callback, *args, **kwargs):
        """

        :param img: an image object <<class PIL.Image.Image>>
        :param callback: method
        :param args:
        :param kwargs:
        :return: x, y, zV, t; np.ndarray-s with x, y, z coordinates of the particles and execution time
        """
        x, y, zV, t = np.array([]), np.array([]), np.array([]), np.array([])
        for page in PIL.ImageSequence.Iterator(img):
            start = time.time()
            im = np.array(page)
            if im.ndim == 3:
                im = im[:, :, 0]
            x_i, y_i, z_i = callback(im, *args, **kwargs)
            end = time.time()
            x = np.append(x, x_i)
            y = np.append(y, y_i)
            zV = np.append(zV, z_i)
            t = np.append(t, end - start)
        return x, y, zV, t

    @classmethod
    def _process(
            cls,
            filename: str,
            thrtype: str,
            thr: float,
            R: float,
            roi: list,
            dz: float,
            z0: int,
            z_pos: bool,
            draw: bool,
            callback: Callable,
            **kwargs
            ):
        """

        :param filename: string; path to the image file
        :param thrtype: string; 'topfraction' or 'topvalue'; type of threshold to apply
        :param thr: non-negative float; threshold value
        :param R: float > 0.5; radius of window filter
        :param roi: list with ints of length 4; region of interest to select
        :param dz: float; z step between images
        :param z0: int; first image position
        :param z_pos: bool; whether to calculate z position
        :param draw: bool; whether to show a drawing
        :param callback: function
        :param kwargs:
        :return: x, y, zV, t; np.ndarray-s with x, y, z coordinates of the particles and execution time
        """
        img = PIL.Image.open(filename)
        roi = cls._normalize_roi_for_image(roi, img)

        # Precalculate matrices for local gradient calculations
        matrices = local_gradient_alloc(img_sz=(roi[1] - roi[0] + 1, roi[3] - roi[2] + 1), R=abs(R))

        x, y, zV, t = cls._process_image(
            img,
            callback,
            thrtype,
            thr,
            R,
            matrices,
            roi,
            z_pos,
            #img=img,
            #callback=callback,
            **kwargs
        )
        if draw:
            cls._draw(z0, dz, zV, x, y, t)
        return x, y, zV, t

    @staticmethod
    def _draw(z0, dz, zV, x, y, t):
        """

        :param z0: int; first image position
        :param dz: float; z step between images
        :param zV: np.ndarray; z coordinates of the particles
        :param x: np.ndarray; x coordinates of the particles
        :param y: np.ndarray; y coordinates of the particles
        :param t: np.ndarray; execution time
        :return: None
        """
        # apply polynomial fit to z - value
        xplot = np.arange(z0, z0 + dz * zV.shape[0], dz)
        if not zV.shape == xplot.shape:
            print(""" Shape of the zV and xplot parameters passed to numpy.polyfit function must be the same. 
                      Сheck if the z0, dz, zV parameters are correct.
                      This method is useful in case of image sequence processing. """)
        else:
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

            fig.update_layout(height=600, width=1400, title_text="Astigmatism xyz")
            fig.show()

    @classmethod
    @validate
    def get_pos_astigmatism(
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

                CALCULATE X, Y, Z COORDINATES OF FLUORESCENT PARTICLE
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
        x, y, zV, t = cls._process(
            filename,
            thrtype,
            thr,
            R,
            roi,
            dz,
            z0,
            z_pos,
            draw,
            callback=get_position_astigmatism,
            positiveAngle=positiveAngle
        )
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
        """roi

                CALCULATE X, Y, Z COORDINATES OF PARTICLE IN BRIGHTFIELD MICROSCOPE
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

    @classmethod
    @validate
    def get_pos_doublehelix(
            cls,
            filename: str,
            thrtype: str,
            thr: float,
            R: float,
            mid_rng: int,
            roi: list = [0, 0, 0, 0],
            dz: float = 0.02,
            z0: int = -1,
            z_pos: bool = True,
            draw: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """

                CALCULATE X, Y, Z COORDINATEs OF THE FLUORESCENT PARTICLE IN DOUBLE-HELIX-BASED MICROSCOPY
        :param filename: string; path to the image file
        :param thrtype: string; 'topfraction' or 'topvalue'
        :param thr: non-negative float; threshold level
        :param R: float > 0.5; radius of window filter
        :param mid_rng: int; indication of mid-range angle of rotation [-180..180] (measured from positive x axis in
           counter-clockwise direction)
        :param z_pos: bool; whether to calculate z position
        :param draw: bool; whether to show a drawing
        :return: tuple with 3 floats; x,y,z coordinate particle center
        """
        x, y, zV, t = cls._process(
            filename,
            thrtype,
            thr,
            R,
            roi,
            dz,
            z0,
            z_pos,
            draw,
            callback=get_position_doublehelix,
            mid_rng=mid_rng
        )
        return x, y, zV, t


    @classmethod
    def _draw_trajectory(
            cls,
            xy,
            fst_im
    ):
        """
            DRAW TRAJECTORY FOR EACH PARTICLE ON FIRST IMAGE FRAME BACKGROUND
        :param xy: 3d np.ndarray; shape=(num_of_frames, x_y_coord=2, num_of_trajectories)
        :param fst_im: 2d np.ndarray; first image frame
        :return: None
        """
        f = plt.figure()
        f.set_figwidth(15)
        f.set_figheight(15)

        for p in range(xy.shape[2]):
            plt.plot(xy[:, 0, p], xy[:, 1, p], color="r", linewidth=1)
        plt.imshow(fst_im)
        plt.show()


    @classmethod
    @validate
    def get_multi_trajectory(
            cls,
            filename: str,
            thrtype: str,
            thr: float,
            R: float,
            epsilon: float,
            minpts: int,
            draw_traject: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """

                LINKS POSITION DATA INTO TRAJECTORIES
        :param filename: string; path to the image file
        :param thrtype: string; 'topfraction' or 'topvalue'
        :param thr: non-negative float; threshold level
        :param R: float > 0.5; radius of window filter
        :param epsilon: float > 0; neighborhood search radius: The maximum distance between two samples for one to be
                        considered as in the neighborhood of the other (DBSCAN)
        :param minpts: int > 0; minimum number of neighbors minpts required to identify a core point
        :param draw_traject: bool
        :return: np.ndarray-S; trj_id, frames, xy, trj_num; shapes: (num_of_trajectories, num_of frames),
                 (num_of_trajectories, num_of frames), (num_of_trajectories, num_of frames, 2), (num_of_trajectories, )
        """
        img_lst = cls._multiframeimg_to_arr(filename)
        fst_im = img_lst[0]

        lgm_partial = partial(
            local_gradient_multi, R=R, epsilon=epsilon, minpts=minpts, thrtype=thrtype, thr=thr
        )
        coord = cls._parallel(lgm_partial)(img_lst)
        trj_id, frames, xy, trj_num = detect_trj(coord, dc=10, dfr=4, Nfr_min=10)

        if draw_traject:
            cls._draw_trajectory(xy, fst_im)
        return trj_id, frames, xy, trj_num