"""
An example of using ParticleDetector class methods to determine the coordinates of a particle in fluorescent
and brightfield microscopes.

Authors: Oleksandr Perederii, Anatolii Kashchuk
2022
"""

from particle_detector import ParticleDetector
import os



if __name__ == "__main__":
    path = os.getcwd()

    """Calculate x, y, z coordinates of the particle in brightfield microscope. """
    x_bright, y_bright, z_bright = ParticleDetector.get_pos_brightfield(
        filename=os.path.join(path, "test_images/Brightfield/Im000.bmp"),
        thrtype="topfraction",
        thr=2.0,
        R=25.0,
        z_pos=True,
        draw=False
    )

    print("x_bright, y_bright, z_bright: ", x_bright, y_bright, z_bright)

    """Calculate x, y, z coordinates of fluorescent particle in astigmatism-based microscopy. First frame from the 
       sequence of 101 particle images is used as input. """
    x_fluor, y_fluor, z_fluor, t_fluor = ParticleDetector.get_pos_astigmatism(
        filename=os.path.join(path, "test_images/Astigmatism/Im_001.png"),
        thrtype='topfraction',
        thr=2.0,
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

    """Calculate x, y, z coordinates of the particle in fluorescent microscope. Sequence of 101 particle images is 
       used as input.
    """
    x_fluor_multi, y_fluor_multi, z_fluor_multi, t_fluor_multi = ParticleDetector.get_pos_astigmatism(
        filename=os.path.join(path, "test_images/Astigmatism_mult_frame/calib_bead_14_MMStack_Pos0.ome.tif"),
        thrtype='topfraction',
        thr=2.0,
        R=12.5,
        positiveAngle=90,
        roi=[190, 289, 226, 325],
        dz=0.02,
        z0=-1,
        z_pos=True,
        draw=True
    )
    print("x_fluor_multi: ", x_fluor_multi)
    print("y_fluor_multi: ", y_fluor_multi)
    print("z_fluor_multi: ", z_fluor_multi)
    print("t_fluor_multi: ", t_fluor_multi)

    """Calculate x, y, z coordinates of fluorescent particle in double-helix microscopy. Sequence of 101 particle
       images is used as input. """
    x_dh, y_dh, z_dh, t_dh = ParticleDetector.get_pos_doublehelix(
        filename=os.path.join(path, "test_images/Double_helix/z_dh_001.png"),
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

    """Calculate x, y coordinates for multiple particles from multiframe image, detect trajectory for each particle and
       plot them on the background of the first image frame"""
    _, _, xy, _ = ParticleDetector.get_multi_trajectory(
        filename=os.path.join(path, "test_images/Multiparticle/Multi_Particle_Stack.tif"),
        thrtype="topfraction",
        thr=3.,
        R=3.,
        epsilon=3,
        minpts=10,
        dc=10,
        dfr=4,
        Nfr_min=10,
        draw_traject=True
    )
    print("x coordinates of particles coordinates on a series of successive frames: ", xy[:, :, 0])
    print("y coordinates of particles coordinates on a series of successive frames: ", xy[:, :, 1])
