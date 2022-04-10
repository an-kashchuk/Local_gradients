function [x,y,z,Gthr] = xyz_dh_express(Im,R,thrsize,mid_rng,varargin)
%XYZ_DH_EXPRESS calculates xyz positions of the particle in astigmatism-based 
% microscopy from its image with a single function using local gradient method.
% [x,y,z,Gthr] = xyz_dh_express(Im,R,thrsize,mid_rng,thrtype)
%   INPUT:
%       Im - input image
%       R - radius of the window (should be >0.5)
%       thrsize - threshold value
%       mid_rng - indication of mid-range angle of rotation [-180..180]
%       thrtype(optional) - type of threshold to apply: 
%           'topfraction' - sets threshold as max_intensity/thr, e.g., for
%               a thr=2 the threshold will be set to half the maximum pixel
%               value
%           'topvalue' - sets the threshold to thr
% 
%   OUTPUT:
%       x,y,z - calculated positions of the particle
%       Gthr - thresholded image
%
% Author: Anatolii Kashchuk


if isempty(varargin)
    thrtype='topfraction';
else
    thrtype=varargin{1};
end

% Precalculate matrices for local gradient calculations
[GMatxfft,GMatyfft,Smatfft] = LocalGradient.local_gradient_alloc(size(Im),R);

% Calculate local gradients images
[~,Gx,Gy,Gthr,~,lsq_data] = LocalGradient.local_gradient(Im,R,GMatxfft,GMatyfft,Smatfft,thrtype,thrsize);

%% Find X-Y position using least square intersection point of gradient lines
[cx,cy] = LocalGradient.lstsqr_lines(lsq_data{1},lsq_data{2},lsq_data{3});

% Correct determined positiones for the reduction in the image size
cR=ceil(R-0.5);
x=cx+cR;
y=cy+cR;

%% Calculate Z value if requested
if nargout>=3
    z=LocalGradient.z_dh(lsq_data{1},mid_rng);
end