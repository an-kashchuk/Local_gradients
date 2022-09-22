function xyarray = local_gradient_multi(Im,R,thr,thrtype,epsilon,minpts)
%local_gradient_multi calculates z-value for a fluorescent particle in astigmatism-based microscopy
% [zV,zVx,zVy] = z_value(Gx,Gy,cx,cy)
%   INPUT:
%       Im - input image
%       R - radius of the window (should be >0.5)
%       thr - threshold value
%       thrtype - type of threshold to apply: 
%           'topfraction' - sets threshold as max_intensity/thr, e.g., for
%               a thr=2 the threshold will be set to half the maximum pixel
%               value
%           'topvalue' - sets the threshold to thr
%       epsilon - neighborhood search radius (see DBSCAN) 
%       minputs -  minimum number of neighbors minpts required to identify a core point (see DBSCAN) 
% 
%   OUTPUT:
%       xyarray - [Nx2] array of xy coordinates of the detected particles,
%                 where N is the number of particles
% 
% Author: Anatolii Kashchuk
% 
% See also DBSCAN

% calculate local gradients
[GMatxfft,GMatyfft,Smatfft] = LocalGradient.local_gradient_alloc(size(Im),R); 
[G,~,~,~,Gmask,lsq_data] = LocalGradient.local_gradient(Im,R,GMatxfft,GMatyfft,Smatfft,thrtype,thr);
% figure,imshow(G,[])
% figure,imshow(Gmask,[])

% cluster data 
idx=dbscan(lsq_data{1},epsilon,minpts);

coord=zeros(max(idx),2);
% go through all detected clusters
for i=1:max(idx)
    indx=idx==i; % get index vector
    [cx,cy] = LocalGradient.lstsqr_lines(lsq_data{1}(indx,:),lsq_data{2}(indx,:),lsq_data{3}(indx)); % find centers
    coord(i,:)=[cx,cy];
end

%% Output
xyarray=coord+ceil(R-0.5);
end