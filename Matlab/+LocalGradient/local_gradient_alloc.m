function [GMatxfft,GMatyfft,Smatfft] = local_gradient_alloc(Imsz,R)
%LOCAL_GRADIENT_ALLOC preallocates matrices for gradient_filter.
%   [GMatxfft,GMatyfft,Smatfft] = local_gradient_alloc(Imsz,R)
% 
%   INPUT:
%       Imsz - size of the image for which local gradient is calculated
%       R - radius of the window (should be >0.5)
% 
%   OUTPUT:
%       GMatxfft,GMatyfft,Smatfft - 2D fourier transform matrices for
%       calculation of horizontal, vertical gradients and sum of pixels
%       correspondingly
% 
% Author: Anatolii Kashchuk

cR=ceil(R-0.5);
h = fspecial('disk',R);
h=h/max(h(:));

[GMatx,GMaty] = meshgrid(-cR:cR);

outsz1=Imsz(1)+2*cR+1;
outsz2=Imsz(2)+2*cR+1;
GMatxfft=fft2(GMatx.*h,outsz1,outsz2);
GMatyfft=fft2(GMaty.*h,outsz1,outsz2);
Smat = h;
Smatfft=fft2(Smat,outsz1,outsz2);
