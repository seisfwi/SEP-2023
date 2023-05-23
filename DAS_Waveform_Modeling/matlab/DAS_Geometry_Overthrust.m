%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Define the discreate point of the fiber and finish related computation
%
%   Author: Haipeng Li
%   Date  : 2023/05/01 
%   Email : haipeng@stanford.edu
%   Affiliation: SEP, Stanford University
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear;clc;close all;

%% Define parameters
nx = 241;
nz = 101;
dx = 10;
x = [0:nx-1]*dx;
z = [0:nz-1]*dx;
vp = ones(nz, nx) * 4000;
vp( 75:end, :) = 4500;
vp(150:end, :) = 5000;
offset = 1000;    
nq = 5;

%% define the control points of the DAS fiber
P = zeros(20, 3);
P(1, :)  = [0.5, 0.00, 0];
P(2, :)  = [0.5, 0.10, 0];
P(3, :)  = [0.5, 0.20, 0];
P(4, :)  = [0.5, 0.30, 0];
P(5, :)  = [0.5, 0.40, 0];
P(6, :)  = [0.5, 0.50, 0];
P(7, :)  = [0.5, 0.60, 0];
P(8, :)  = [0.5, 0.70, 0];
P(9, :)  = [0.6, 0.74, 0];
P(10, :) = [0.7, 0.75, 0];
P(11, :) = [0.8, 0.75, 0];
P(12, :) = [0.9, 0.75, 0];
P(13, :) = [1.0, 0.75, 0];
P(14, :) = [1.2, 0.75, 0];
P(15, :) = [1.5, 0.75, 0];
P(16, :) = [1.6, 0.75, 0];
P(17, :) = [1.7, 0.75, 0];
P(18, :) = [1.8, 0.75, 0];
P(19, :) = [1.9, 0.75, 0];
P(20, :) = [2.0, 0.75, 0];

P  = P * 1000; % km to m
Px = P(:,1);
Py = P(:,2);
Pz = P(:,3);


%% DAS points
GL = 10;
nq = 5;  % nq must be an odd number

% get equally spaced segements      
[PI, error] = segment(P, GL);
npts = size(PI, 1);

%% DAS quadrature points
npts_quad = npts + (npts-1) * (nq-2);
PI_Quad = interparc(npts_quad, Px, Py, Pz, 'spline');

% compute the Frenet-Serret
[T, N, B, k, t] = frenet(PI_Quad(:,1), PI_Quad(:,2));


%% Compute the weights for integral
weights = zeros(npts, 6); 

X = [1, 0, 0];
Y = [0, 1, 0];
Z = [0, 0, 1];

% For the first and last points, weights for: exx, exy, exz, eyy, eyz, ezz
weights(1, 1) = dot(T(1, :), X)^2;
weights(1, 2) = dot(T(1, :), X) * dot(T(1, :), Y) * 2;
weights(1, 3) = dot(T(1, :), X) * dot(T(1, :), Z) * 2;
weights(1, 4) = dot(T(1, :), Y)^2;
weights(1, 5) = dot(T(1, :), Y) * dot(T(1, :), Z) * 2;
weights(1, 6) = dot(T(1, :), Z)^2;

weights(npts, 1) = dot(T(npts_quad, :), X)^2;
weights(npts, 2) = dot(T(npts_quad, :), X) * dot(T(npts_quad, :), Y) * 2;
weights(npts, 3) = dot(T(npts_quad, :), X) * dot(T(npts_quad, :), Z) * 2;
weights(npts, 4) = dot(T(npts_quad, :), Y)^2;
weights(npts, 5) = dot(T(npts_quad, :), Y) * dot(T(npts_quad, :), Z) * 2;
weights(npts, 6) = dot(T(npts_quad, :), Z)^2;

% For those points in the middle, weights for: exx, exy, exz, eyy, eyz, ezz
for i = 2 : npts-1
    for j = 1 : nq
        q = (i-1) * (nq-1) + j - (nq-1)/2;
        weights(i, 1) = weights(i, 1) + dot(T(q, :), X) * dot(T(q, :), X);
        weights(i, 2) = weights(i, 2) + dot(T(q, :), X) * dot(T(q, :), Y) * 2;
        weights(i, 3) = weights(i, 3) + dot(T(q, :), X) * dot(T(q, :), Z) * 2;
        weights(i, 4) = weights(i, 4) + dot(T(q, :), Y) * dot(T(q, :), Y);
        weights(i, 5) = weights(i, 5) + dot(T(q, :), Y) * dot(T(q, :), Z) * 2;
        weights(i, 6) = weights(i, 6) + dot(T(q, :), Z) * dot(T(q, :), Z);
    end
    weights(i, :) = weights(i, :) * GL / nq / GL;
end 


% data to save: coord_x, coord_y, coord_z, exx, exy, exz, eyy, eyz, ezz
data = [PI, weights];
save(sprintf("DAS_cable_par_%.2fm.dat", GL), "data", "-ascii")

% plot weights
figure(1)
hold on;
plot(weights(:, 1))  
plot(weights(:, 2))
plot(weights(:, 4))
legend('coefficient_exx', 'coefficient_exy', 'coefficient_eyy')


% plot geometry
figure(2); hold on;
pcolor(x,z,vp); colormap("gray"); colorbar; clim([2000, 6000]); shading interp;
scatter(Px , Py , 'r*');
scatter(PI(:,1), PI(:,2), 'bo');
xlim([x(1), x(end)]);
ylim([z(1), z(end)]);
set(gca, 'YDir', 'reverse', 'FontSize', 16);
xlabel('Distance (m)', 'FontSize', 16);
ylabel('Depth (m)', 'FontSize', 16);


