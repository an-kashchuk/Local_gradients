% This example shows how to use Local Gradient software to determine x,y,z
% position of a fluorescent particle in astigmatism-based microscopy.
%
% Author: Anatolii Kashchuk

close all
clearvars

%% Input parameters

fname='../test_images/Fluorescent'; % path to multipage tif file

% Set local gradient parameters
R=14; % set window size
thrsize=2; % threshold value
thrtype= 'topfraction'; % threshold type

% Set angle in degrees of positive direction of the particle's image (measured from
% positive x axis in counter-clockwise direction
PositiveAngle=90;

% Set axial displacement parameters (if known) - for plots only
dz=0.020; % z step between images
z0=-1; % first image position

%% Analysis

% Get number of images in the folder
numimgs = 101;

% Get image size
ImI=imfinfo(fullfile( fname,'Im_001.png'));
imsize = [ImI.Height, ImI.Width];

% Precalculate matrices for local gradient calculations
[GMatxfft,GMatyfft,Smatfft] = LocalGradient.local_gradient_alloc(imsize,R);


% Preallocate arrays
zV=zeros(numimgs,1);
t=zeros(numimgs,1);
x=zeros(numimgs,1);
y=zeros(numimgs,1);
% Go through all the images
for i=1:numimgs
    % Read image
    Im=double( imread( fullfile( fname,['Im_' num2str(i,'%.3d') '.png'] ) ) );
    figure(10),imshow(Im,[])
    
    tic
    
    % Calculate local gradients images
    [G,Gx,Gy,Gthr,Gmask,lsq_data] = LocalGradient.local_gradient(Im,R,GMatxfft,GMatyfft,Smatfft,thrtype,thrsize);
    
    % Find center of symmetry of the particle
    [cx,cy] = LocalGradient.lstsqr_lines(lsq_data{1},lsq_data{2},lsq_data{3});

    % Correct determined positiones for the reduction in the image size
    cR=ceil(R-0.5);
    x(i)=cx+cR;
    y(i)=cy+cR;
    
    % Calculate z-value
    zV(i) = LocalGradient.z_fluor(Gthr,Gx,Gy,cx,cy,PositiveAngle);

    t(i)=toc;

end

%% Apply polynomial fit to z-value
xplot=z0:dz:z0+(numimgs-1)*dz;
[p,S]=polyfit(xplot,zV',4); % fit to 4th degree polynomial
[z_fit,delta] = polyval(p,xplot,S); % get fit error estimate 

%% Plot results

figure(50), plot(xplot,x), grid minor, grid on,hold on
            title('X position'),xlabel('z-position, \mum'), ylabel('x, pxls')
figure(51), plot(xplot,y), grid minor, grid on,hold on
            title('Y position'),xlabel('z-position, \mum'), ylabel('y, pxls')
figure(52), plot(xplot,zV,'.',xplot,z_fit), grid minor, grid on,hold on
            title('Z-value'),xlabel('z-position, \mum'), ylabel('z-value'),legend('data',['fit. Error estimate=' num2str(mean(delta))])
figure(53), plot(xplot,t), grid minor, grid on,hold on
            title(['Execution time, t_{av}=' num2str(mean(t))]),xlabel('z-position, \mum'), ylabel('t, seconds')
            
%% Alternatevily, the position of the particle can be determined with a single express function
% Express calculation (uncomment to use)
% [x,y,z,Gthr]=LocalGradient.xyz_fluor_express(Im,R,thrsize,PositiveAngle,thrtype);