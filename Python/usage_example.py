"""
An example of using ParticleDetector class methods to determine the coordinates of a particle in fluorescent
and brightfield microscopes.

Authors: Oleksandr Perederii, Anatolii Kashchuk
"""

from particle_detector import ParticleDetector
import os

if __name__=="__main__":
    path = os.getcwd()

    """
        Calculate x, y, z coordinates of the particle in brightfield microscope. 
    """
    x_bright, y_bright, z_bright = ParticleDetector.get_pos_brightfield(
        filename=os.path.join(path, "../test_images/Brightfield/Im000.bmp"),
        thrtype="topfraction",
        thr=2.0,
        R=25.0,
        z_pos=True,
        draw=False
    )

    print("x_bright, y_bright, z_bright: ", x_bright, y_bright, z_bright)

    """
        Calculate x, y, z coordinates of the particle in fluorescent microscope. Sequence of 101 particle images is 
        used as input.
    """
    x_fluor, y_fluor, z_fluor, t_fluor = ParticleDetector.get_pos_fluorescent(
        filename=os.path.join(path, "../test_images/Fluorescent/Im_001.png"),
        thr=2.0,
        thrtype='topfraction',
        R=14,
        positiveAngle=90,
        dz=0.02,
        z0=-1,
        z_pos=True,
        draw=False
    )
    print("x_fluor: ", x_fluor)
    print("y_fluor: ", y_fluor)
    print("z_fluor: ", z_fluor)
    print("t_fluor: ", t_fluor)
