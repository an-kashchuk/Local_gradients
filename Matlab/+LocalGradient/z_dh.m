function zV = z_dh(pnts,mid_rng)
%z_dh calculates z-value for a fluorescent particle in double-helix-based microscopy
%   zV = z_dh(pnts,mid_rng)
%       pnts - coordinates of points in the image (2-column vector)
%       mid_rng - indication of mid-range angle of rotation [-180..180]
% 
% Author: Anatolii Kashchuk

x=pnts(:,1);
y=-pnts(:,2);

% Calculate moments
M00=numel(x);
M10=sum(x);
M01=sum(y);
M11=sum(x.*y);
M20=sum(x.^2);
M02=sum(y.^2);

% Find center
cx=M10/M00;
cy=M01/M00;

% Calculate central moments
u11=M11/M00-cx*cy;
u20=M20/M00-cx^2;
u02=M02/M00-cy^2;

% Calculate the angle
theta=0.5*atan2d(2*u11,(u20-u02));

%% Correct angle to the specified range
fromRng=mid_rng-90;
toRng=mid_rng+90;
if theta < fromRng
    theta = theta+180;
elseif theta > toRng
    theta = theta-180;
end

%% Output
zV=theta;
end
