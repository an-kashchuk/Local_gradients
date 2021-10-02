function [cx,cy,P,dR,x,p,l] = lstsqr_lines(P1,P2,w)
%LSTSQR_LINES least-square line intersection
%   [cx,cy,P,dR,x,p,l] = lstsqr_lines(P1,P2,w)
% 
%      INPUT:  
%       P1 - 2xN array of first points that define lines
%       P2 - 2xN array of second points that define lines
%       w - line weights
% 
%      OUTPUT:
%       cx,cy - x,y coordinates of the intersection point
%       P - coordinates of nearest points on each line
%       dR - distance from intersection to each line
%       x - intersection point in a cell format for plotting
%       p - same as P but in a cell format
%       l - input lines in a cell format for plotting
% Algorithm adapted from: 
% 1. Serge (2021). Line-Line Intersection (N lines, D space), MATLAB Central File Exchange.
% 2. Johannes Traa (2013) - Least-Squares Intersection of Lines
% 
% Author: Anatolii Kashchuk

n = P2 - P1; %vectors from A to B
n = n./sqrt(sum(n.^2,2)); %normalized vectors
[r,c] = size(P1);
Inn=n'.*reshape(n,[1 r c])-reshape(eye(c),[c 1 c]); % columns = lines, rows
Inn=Inn.*w;
dR=reshape(sum(Inn,2),[c c]);
q=reshape(Inn,[c r*c])*P1(:);

C=lsqminnorm(dR,q);
cx=C(1);
cy=C(2);

%extra outputs
if nargout>2
    U = sum((C'-P1).*n,2);
    P = P1 + U.*n; %nearest point on each line
end
if nargout>3
    dR = sqrt(sum(bsxfun(@minus,C',P).^2,2)); %distance from intersection to each line
end
 
%plot outputs
if nargout>4
    x = num2cell(C); %intersection point X
end
if nargout>5
    p = num2cell(P,1); %tangent points P
end
if nargout>6
    l = mat2cell([P1(:) P2(:)]',2,ones(1,c)*r); %initial lines A,B using cell format {x y..}
end
end
