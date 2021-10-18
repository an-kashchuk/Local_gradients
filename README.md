# Local_gradients
This package provides a set of tools for 3-D localisation of single particles in brightfield and fluorescent microscopy using local gradients.
The package is provided in LabVIEW, Matlab and Python and have been tested to provide the same result (within the rounding errors). Please, note that the calculated position of the particle in xy is defined accourding to the array indexing convenience in the specific language (i.e. the position from LabVIEW will be 1 pixel smaller than the result from Matlab for the same image)

## Usage

For the usage of the package please see the examples provided for each language. Examples were preset to run the images in [test_images](https://github.com/an-kashchuk/Local_gradients/tree/main/test_images) folder

### Matlab
The easiest way to calculate the position of the particle is from xyz_express.m and xyz_fluor_express.m (for brightfield and fluorescent images correspondingly).
For more efficient calculations see examples.

**Requirements:**  
*Matlab (tested for 2019b)  
Image Processing Toolbox*

### LabVIEW
The easiest way to calculate the position of the particle is from SubVI/xyz-express.vi and SubVI/xyz-fluor_express.vi (for brightfield and fluorescent images correspondingly).
For more efficient calculations see examples.

**Requirements:**  
*Labview 2015 SP1 or newer (tested on 2015 SP1 only)  
Vision Acquisition Software 2016 (for example file only)*


### Python
local_gradient_math.py contains all the necessary methods for particle localization.

**Required libraries:**  
*PIL==7.1.2  
cv2==4.1.2  
matplotlib==3.2.2  
numpy==1.19.5  
plotly==4.4.1  
scipy==1.4.1*  

## License
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
If you use this package, please, cite us as:
