function [x,y,z,Gthr] = xyz_brt_express(Im,R,thrsize,varargin)
%XYZ_BRT_EXPRESS calculates xyz positions of the particle from its image with a single function using local gradient method.
% [x,y,z,Gthr] = xyz_brt_express(Im,R,thrsize,thrtype)
%   INPUT:
%       Im - input image
%       R - radius of the window (should be >0.5)
%       thrtype(optional) - type of threshold to apply: 
%           'topfraction' - sets threshold as max_intensity/thr, e.g., for
%               a thr=2 the threshold will be set to half the maximum pixel
%               value (default)
%           'topvalue' - sets the threshold to thr
%       thrsize - threshold value
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

[GMatxfft,GMatyfft,Smatfft] = LocalGradient.local_gradient_alloc(size(Im),R);
[~,Gx,Gy,Gthr,~,lsq_data] = LocalGradient.local_gradient(Im,R,GMatxfft,GMatyfft,Smatfft,thrtype,thrsize);

%% Find X-Y position using least square intersection point of gradient lines
[cx,cy] = LocalGradient.lstsqr_lines(lsq_data{1},lsq_data{2},lsq_data{3});

% Correct for change in size of the image
cR=ceil(R-0.5);
x=cx+cR;
y=cy+cR;

%% Calculate Z value if requested
if nargout>=3
    z=LocalGradient.z_brt(Gx,Gy,cx,cy);
end
end

