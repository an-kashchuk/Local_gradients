function [G,Gx,Gy,Gthr,Gmask,lsq_data] = local_gradient(Im,R,GMatxfft,GMatyfft,Smatfft,thrtype,thr)
%local_gradient calculates local gradients of the image
% [G,Gx,Gy,Gthr,Gmask,lsq_data] = local_gradient(Im,R,GMatxfft,GMatyfft,Smatfft,thrtype,thr)
% 
%   INPUT:
%       Im - input image
%       R - radius of the window (should be >0.5)
%       GMatxfft,GMatyfft,Smatfft - 2D fourier transform matrices for 
%           calculation of horizontal, vertical gradients and sum of pixels 
%           correspondingly
%       thrtype - type of threshold to apply: 
%           'topfraction' - sets threshold as max_intensity/thr, e.g., for
%               a thr=2 the threshold will be set to half the maximum pixel
%               value
%           'topvalue' - sets the threshold to thr
%       thr - threshold value
% 
%   OUTPUT:
%       G - magnitude of local gradients
%       Gx,Gy - horizontal and vertical local gradients
%       Gthr - thresholded image
%       Gmask - mask extracted from the thresholded image
%       lsq_data - formatted data for least square fit of intersection of
%           lines: {pixel_centers,pixel_gradients,gradient_magnitude} (see 
%           lstsqr_lines)
% 
% Author: Anatolii Kashchuk

cR=ceil(R-0.5);
outsz=[2*cR+1,size(Im,1);2*cR+1,size(Im,2)];
Im=double(Im)+1;  % Add 1 to avoid division by 0

Imfft=fft2(Im,outsz(1,1)+outsz(1,2),outsz(2,1)+outsz(2,2)); % FFT of input image

%% Sum of pixels in the area
ImSum=ifft2(Imfft.*Smatfft);
ImSum=ImSum(outsz(1,1):outsz(1,2),outsz(2,1):outsz(2,2)); % Select valid area
% figure(50),imshow(ImSum,[])
%% X gradient
Gx=ifft2(Imfft.*GMatxfft);
Gx=Gx(outsz(1,1):outsz(1,2),outsz(2,1):outsz(2,2))./ImSum;

%% Y gradient
Gy=ifft2(Imfft.*GMatyfft);
Gy=Gy(outsz(1,1):outsz(1,2),outsz(2,1):outsz(2,2))./ImSum;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
G=sqrt((Gx).^2+(Gy).^2);  % gradient magnitude

switch thrtype
    case 'topfraction'
        cond=max(G(:))/thr;
    case 'topvalue'
        cond=thr;
    otherwise
        error('condition type: "topfraction"/"topvalue" or  is not specified ')
end

if nargout==4 || nargout==5
    Gmask=imbinarize(G,cond);
%     Gmask = bwareafilt(Gmask,1); % keep only the largest region
    Gthr=Gmask.*G;
elseif nargout==6
    Gsize=size(G); % size of the image
    [row,col] = find(G>cond);  % find non-zero elements only
    
    Gmask=zeros(Gsize); 
    Gmask(G>cond)=1; % return mask
    
%     Gmask = bwareafilt(logical(Gmask),1); % keep only the largest region
    
    Gthr=Gmask.*G; % return thresholded image
    
    origins=[col,row];  % centers of pixels are first gradient line points
    grad=[(Gx(sub2ind(Gsize,row,col))+col),(Gy(sub2ind(Gsize,row,col))+row)];  % create array of [Gx, Gy] points
    lind=sub2ind(Gsize,row,col);
    v=G(lind); % Magnitude matrix to a vector
    
    lsq_data={origins,grad,v',lind}; % data for least square fitting
end
end

