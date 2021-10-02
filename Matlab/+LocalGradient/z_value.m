function [zV,zVx,zVy] = z_value(Gx,Gy,cx,cy)
%z_value calculates z-value for an image of a particle
% z-value is calculated as the difference of sums of all local gradients in
% the image of a particle splitted horizontally and vertically by the
% center of the particle.
% 
% [zV,zVx,zVy] = z_value(Gx,Gy,cx,cy)
%   INPUT:
%       Gx,Gy - horizontal and vertical local gradients
%       cx,cy - x- and y-coordinates of the center of the particle
% 
%   OUTPUT:
%       zV - z-value
%       zVx, zVy - z-value from horizontal and vertical split.
%       Gthr - thresholded image
%       Gmask - mask extracted from the thresholded image
%       lsq_data - formatted data for least square fit of intersection of
%           lines: {pixel_centers,pixel_gradients,gradient_magnitude} (see 
%           lstsqr_lines)
% 
% Author: Anatolii Kashchuk

%% %%% X - horizontal split

% find central column
ccol=ceil(cx-0.5);

% split to left and right parts
GxLeft=Gx(:,1:ccol-1);
GxRight=Gx(:,ccol+1:end);

% find sum of central column and corresponding fractions of a pixel
GxCentrSum=sum(Gx(:,ccol));
frxL=cx-(ccol-0.5);
frxR=1-frxL;

% sum all the elements in both parts
GxLSum=sum(GxLeft(:))+GxCentrSum*frxL;
GxRSum=sum(GxRight(:))+GxCentrSum*frxR;

% calculate z value from x gradient
zVx=GxRSum-GxLSum;

%% %%% Y - vertical split

% find central row
crow=ceil(cy-0.5);

% split to top and bottom parts
GyTop=Gy(1:crow-1,:);
GyBottom=Gy(crow+1:end,:);

% find sum of central row and corresponding fractions of a pixel
GyCentrSum=sum(Gy(crow,:));
fryT=cy-(crow-0.5);
fryB=1-fryT;

% sum all the elements in both parts
GyTSum=sum(GyTop(:))+GyCentrSum*fryT;
GyBSum=sum(GyBottom(:))+GyCentrSum*fryB;

% calculate z value from y gradient
zVy=GyBSum-GyTSum;

%% Output
zV=(zVx+zVy)/2;

end

