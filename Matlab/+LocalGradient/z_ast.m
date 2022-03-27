function [zV,AxMJR,AxMNR] = z_fluor(Gthr,Gx,Gy,cx,cy,angPos)
%z_fluor calculates z-value for a fluorescent particle in astigmatism-based microscopy
% [zV,zVx,zVy] = z_value(Gx,Gy,cx,cy)
%   INPUT:
%       Gx,Gy - horizontal and vertical local gradients
%       cx,cy - x- and y-coordinates of the center of the particle
%       angPos - angle of the major axes (in degrees) of the spot 
%                It is used to discriminate positive and negative displacement 
%                of the particle from the in-focus position define positive 
%                (should be measured from the positive direction of x-axis) 
% 
%   OUTPUT:
%       zV - z-value
%       AxMJR, AxMNR - major and minor axes (distances between centers of 
%                      opposite halves of the image - TOP vs BOTTOM, LEFT vs RIGHT)
% 
% The image of local gradients (Gthr) is splitted into 4 parts: 
% TOP, LEFT, BOTTOM, RIGHT, relative to the particle center (cx,cy). The
% least-square intersection of all gradient lines (Gx, Gy) is calculated 
% for each part. These centers are forming two lines: TOP-BOTTOM and LEFT-RIGHT
% which are called axes (similar to ellipses axes). The major of two axes
% represents the magnitude of the z-value. The sign of the z-value is
% determined according to the angPos value which should correspond to the
% position of the long axis of the elliptical image of the particle. angPos
% is not required to be accurate, however, setting it to ~45 degrees off
% the real angle will create sporadic change in the sign and, therefore, in
% z-value.
% 
% Author: Anatolii Kashchuk

Gsize=size(Gthr); % size of the image
%% %%% X - horizontal split

% find central column
ccol=ceil(cx-0.5);

% create horizontally splitted parts of the local gradient image
GthrL=Gthr; % LEFT
GthrR=Gthr; % RIGHT

% find fractions of a pixel for central column
frxL=cx-(ccol-0.5);
frxR=1-frxL;

% split to left and right parts
GthrR(:,1:ccol-1)=0;     GthrR(:,ccol)=GthrR(:,ccol)*frxR;
GthrL(:,ccol+1:end)=0;   GthrL(:,ccol)=GthrL(:,ccol)*frxL;

% figure,imshow(GthrR,[])
% figure,imshow(GthrL,[])
%% %%% Y - vertical split

% find central row
crow=ceil(cy-0.5);

% create vertically splitted parts of the local gradient image
GthrT=Gthr; % TOP
GthrB=Gthr; % BOTTOM

% find fractions of a pixel for central row
fryT=cy-(crow-0.5);
fryB=1-fryT;

% split to left and right parts
GthrT(crow+1:end,:)=0;  GthrT(crow,:)=GthrT(crow,:)*fryT;
GthrB(1:crow-1,:)=0; GthrB(crow,:)=GthrB(crow,:)*fryB;

% figure,imshow(GthrT,[])
% figure,imshow(GthrB,[])

%% % calculate least-square fit of gradient lines for each part
Gz={GthrT,GthrL,GthrB,GthrR};
hX=zeros(1,4);
hY=zeros(1,4);

% loop through each half
for i=1:4
    [row,col] = find(Gz{i}>0);  % find non-zero elements only
    origins=[col,row];  % centers of pixels are first gradient line points
    grad=[(Gx(sub2ind(Gsize,row,col))+col),(Gy(sub2ind(Gsize,row,col))+row)];  % create array of [Gx, Gy] points
    lind=sub2ind(Gsize,row,col); % linear index
    v=Gz{i}(lind); % magnitude matrix to a vector

    lsq_data={origins,grad,v',lind}; % data for least square fitting
    [cxLocal,cyLocal] = LocalGradient.lstsqr_lines(lsq_data{1},lsq_data{2},lsq_data{3});
    
    hX(i)=cxLocal-cx;
    hY(i)=cyLocal-cy;
end

ax1d=sqrt((hX(1)-hX(3)).^2+(hY(1)-hY(3)).^2); % 1st axis length
ax2d=sqrt((hX(2)-hX(4)).^2+(hY(2)-hY(4)).^2); % 2nd axis length

% find major and minor axes length
AxMJR=max([ax1d;ax2d]);
AxMNR=min([ax1d;ax2d]);

%% calculate displacement sign

p1=sind(2*angPos).*( hX(1)-hX(3) )+cosd(2*angPos).*( hY(1)-hY(3));
p2=-cosd(2*(angPos+90)).*( hX(4)-hX(2) )+sind(2*(angPos+90)).*( hY(4)-hY(2));
focus_sign=sign(p1+p2);

%% Output
zV=focus_sign*AxMJR;
end