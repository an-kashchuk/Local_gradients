% This example shows how to use Local Gradient software to determine x,y
% positions of multiple particles in a single image
%
% Author: Anatolii Kashchuk

close all
clearvars

%% Input parameters

% image path
fpath='../test_images/Multiparticle/Multi_Particle_Stack.tif';

R=3; % set window size
thrsize=3; % threshold value
thrtype= 'topfraction'; % threshold type

epsilon=3; % neighborhood search radius (see DBSCAN) 
minpts=10; %  minimum number of neighbors minpts required to identify a core point (see DBSCAN) 
dc=10; % maximum distance from the detected particle to look for linked particles in other frames
dfr=3; % number of frames to look for linked particles
Nfr_min=10; % minimum number of frames in trajectory

%%
% read images 
N=100;
for i=1:N
    Imstack(:,:,i)=double(imread(fpath,i));    
end

% start localization
tic
coord{N,1}=[];
for i=1:N
    Im=Imstack(:,:,i);
    xyarray = LocalGradient.local_gradient_multi(Im,R,thrsize,thrtype,epsilon,minpts);
    coord{i}=xyarray;
end
t=toc/N;
disp(['Average execution time per image: ', num2str(t*1000,'%.1f'), 'ms' ])

% Link positions into trajectories
T_fr = LocalGradient.detect_trj(coord,dc,dfr,Nfr_min);

%% Show an image with detected particles and trajectories
Im_N = 1; % image number to show
figure,
imshow(Imstack(:,:,Im_N),[]), colormap(parula), hold on % show image
plot(coord{Im_N}(:,1),coord{Im_N}(:,2),'ro', 'MarkerSize',20), colorbar % plot circles around particles
for i=1:10
    plot(T_fr.xy{i}(:,1),T_fr.xy{i}(:,2),'-',"Color","#7E2F8E") % add trajectories
end