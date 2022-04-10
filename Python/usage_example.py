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

    """Calculate x, y, z coordinates of fluorescent particle in astigmatism-based microscopy. Sequence of 101 
    particle images is used as input. """
    x_fluor, y_fluor, z_fluor, t_fluor = ParticleDetector.get_pos_astigmatism(
        filename=os.path.join(path, "../test_images/Astigmatism/Im_001.png"),
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

    """Calculate x, y, z coordinates of fluorescent particle in double-helix microscopy. Sequence of 101 particle 
    images is used as input. """
    x_dh, y_dh, z_dh, t_dh = ParticleDetector.get_pos_doublehelix(
        filename=os.path.join(path, "../test_images/Double_helix/z_dh_001.png"),
        thr=1.7,
        thrtype='topfraction',
        R=10,
        mid_rng=91,
        dz=5.,
        z0=180,
        z_pos=True,
        draw=False
    )

    print("x_dh: ", x_dh)
    print("y_dh: ", y_dh)
    print("z_dh: ", z_dh)
    print("t_dh: ", t_dh)