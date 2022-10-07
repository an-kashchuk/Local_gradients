"""
Here are implemented all the util functions of the local gradient method

Authors: Oleksandr Perederii, Anatolii Kashchuk
2022
"""

from multiprocessing import Pool, cpu_count
import numpy as np
import PIL
from functools import partial



def parallel(func):
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


def multiframeimg_to_arr(filename):
    """

            CONVERT MULTIFRAME IMAGE INTO LIST OF ARRAYS
    :param filename: filename: string; path to the image file, may be multi-image
    :return: list that contains image np.ndarray-s
    """
    img = PIL.Image.open(filename)
    img_lst = []

    for i, page in enumerate(PIL.ImageSequence.Iterator(img)):
        img_lst.append(np.array(page))
    return img_lst


def prep_data_for_multi(func, *args, **kwargs):
    """

            CREATE A NEW FUNCTION, A CALLABLE, THAT BEHAVES LIKE INITIAL, BUT HAS LESS ARGUMENTS; USEFULL FOR UTILIZING
            IN multiprocessing.Pool.map
    :param func: callable
    :param args:
    :param kwargs:
    :return: new callable
    """
    return partial(func, *args, **kwargs)




