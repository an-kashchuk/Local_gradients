% This example shows how to use Local Gradient software to calculate local
% gradient images and how to localise objects in the image using least
% square intersection of lines
%
% Author: Anatolii Kashchuk

close all
clearvars

%% Input parameters

% impath='143.68.tif'; % path to the image
impath='../test_images/Brightfield/Im000.bmp'; % path to the image

R=25; % set window size

thrsize=2; % threshold value
thrtype= 'topfraction'; % threshold type

%% Read image

Im=imread(impath);

% average 3 channels if image is RGB
if size(Im,3)~=1
    Im=mean(Im(:,:,1:3),3);
end
figure,imshow(Im,[])

%% Calculate local gradients

[GMatxfft,GMatyfft,Smatfft] = LocalGradient.local_gradient_alloc(size(Im),R); 
% these parameters depend only on the image size and R parameter and can be
% reused for many images if the size and R does not change

[G,Gx,Gy,Gthr,Gmask,lsq_data] = LocalGradient.local_gradient(Im,R,GMatxfft,GMatyfft,Smatfft,thrtype,thrsize);

% Display results
figure,
subplot(2,3,1),imshow(Im,[]),title('Im')
subplot(2,3,2),imshow(Gx,[]),title('Gx'),colormap(gca,jet)
subplot(2,3,3),imshow(Gy,[]),title('Gy'),colormap(gca,jet)
subplot(2,3,4),imshow(G,[]),title('G'),colormap(gca,jet)
subplot(2,3,5),imshow(Gthr,[]),title('Gthr'),colormap(gca,jet)
subplot(2,3,6),imshow(Gmask,[]),title('Gmask')

%% Find X-Y position using least square intersection point of gradient lines
[cx,cy,P,dR,xpts,p,l] = LocalGradient.lstsqr_lines(lsq_data{1},lsq_data{2},lsq_data{3});

% Display gradients
figure,plot(xpts{:},'*k',p{:},'.k',l{:}),axis equal

% Correct for change in size of the image
cR=ceil(R-0.5);
x=cx+cR;
y=cy+cR;

% Display results
figure,imshow(Im,[]), hold on, plot(x,y,'r*'),text(x,y,['  \leftarrow x=' num2str(x),' y=' num2str(y)],'Color', 'r')
disp(['x = ' num2str(x)])
disp(['y = ' num2str(y)])

%% Calculate Z value
z=LocalGradient.z_brt(Gx,Gy,cx,cy);
disp(['z = ' num2str(z)])

%% Alternatevily, the position of the particle can be determined with a single express function
% Express calculation (uncomment to use)
% [x,y,z,Gthr]=LocalGradient.xyz_brt_express(Im,R,thrsize,thrtype);