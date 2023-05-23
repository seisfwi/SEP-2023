%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Generate the quadture points on unit fibers with different curvature
%
%   Author: Haipeng Li
%   Date  : 2023/05/01 
%   Email : haipeng@stanford.edu
%   Affiliation: SEP, Stanford University
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear; clc; clf; close all;

% define gauge length
gl = 10;

% Define the length of the arc segments (in units of length)
l = 1;

% Define the radius of the circles that will generate the arc segments
r = [1  2.0 1e10]/pi;
npts = 101;

fiber_coord = zeros(length(r), 2, npts);

% Generate the x and y coordinates of the arc segments
figure(1);
for i = 1:length(r)
    theta = l/r(i); % calculate the angle of the arc segment
    t = linspace(pi/2 - theta /2, pi/2 + theta /2, npts);
    x = r(i)*cos(t) * gl;
    y = r(i)*sin(t) * gl;
    y = y - y(1);

    % save coord
    fiber_coord(i, 1, :) = x;
    fiber_coord(i, 2, :) = y;

    % plot
    plot(x, y, LineWidth=2)
    hold on
end
axis equal


%% DAS points
% nq = [1, 3, 5, 7, 9, 21];
nq = [1, 3, 9, 21];

X = [1, 0, 0];
Y = [0, 1, 0];
Z = [0, 0, 1];

% get the quadture points
figure(2); hold on
for i = 1 : length(r)
    for j = 1 : length(nq)
        
        q = nq(j);
        quad_coord = zeros(2, q);

        if q == 1
            quad_coord(1, 1) = fiber_coord(i, 1, floor(npts/2)+1);
            quad_coord(2, 1) = fiber_coord(i, 2, floor(npts/2)+1);
        else
            P = [squeeze(fiber_coord(i, :, :))', zeros(npts, 1)];
            PI = interparc(q, P(:, 1), P(:, 2), P(:, 3), 'spline');
            quad_coord(1, :) = PI(:, 1);
            quad_coord(2, :) = PI(:, 2);
        end

        % collect the necessary information
        [T, N, B, k, t] = frenet(quad_coord(1, :), quad_coord(2, :));
        weights = zeros(q, 6);


        for k = 1 : q
            weights(k, 1) = dot(T(k, :), X) * dot(T(k, :), X);
            weights(k, 2) = dot(T(k, :), X) * dot(T(k, :), Y) * 2;
            weights(k, 3) = dot(T(k, :), X) * dot(T(k, :), Z) * 2;
            weights(k, 4) = dot(T(k, :), Y) * dot(T(k, :), Y);
            weights(k, 5) = dot(T(k, :), Y) * dot(T(k, :), Z) * 2;
            weights(k, 6) = dot(T(k, :), Z) * dot(T(k, :), Z);
        end 

        
        data = [quad_coord', zeros(q, 1), weights];
        save(sprintf("Cable%d_quad_%d.dat", i, q), "data", "-ascii");

        subplot(length(r), length(nq), j + (i-1) * length(nq))
        plot(squeeze(fiber_coord(i, 1, :)), squeeze(fiber_coord(i, 2, :)), 'k-', LineWidth=2.5); hold on
        scatter(quad_coord(1, :), quad_coord(2, :), 60, 'MarkerFaceColor',[1, 0, 0])
        xlim([-gl/2-0.8, gl/2+0.8])
        axis equal; set(gca, "FontSize", 20);
        if i == length(r)
            xlabel('Distance (m)')
        end 
        if j == 1
            ylabel('Depth (m)')
        end 

    end
end

% save figure
set(gcf,'position',[5,5,1200,900]);
saveas(gcf,'DAS_Cable.png')


figure(3); hold on
for i = 1 : length(r)
    subplot(length(r), 1, i)
    plot(squeeze(fiber_coord(i, 1, :)), squeeze(fiber_coord(i, 2, :)), 'k-', LineWidth=2.5);
    xlim([-gl/2-0.8, gl/2+0.8])
    axis equal; 
    set(gca,'XTickLabel',[]);
    set(gca,'YTickLabel',[]);
    set(gca,'xtick',[])
    set(gca,'ytick',[])
end 
set(gcf,'position',[5,5,400,300]);
saveas(gcf,'DAS_Cable_Shape.png')




i = 2;
P = [squeeze(fiber_coord(i, :, :))', zeros(npts, 1)];
PI = interparc(q, P(:, 1), P(:, 2), P(:, 3), 'spline');
quad_coord(1, :) = PI(:, 1);
quad_coord(2, :) = PI(:, 2);



figure(4); hold on
for i = 1 : length(r)
    subplot(length(r), 1, i)
    plot(squeeze(fiber_coord(2, 1, :)), squeeze(fiber_coord(2, 2, :)), 'k-', LineWidth=2.5); hold on
    scatter(quad_coord(1, :), quad_coord(2, :), 60, 'MarkerFaceColor',[0, 0, 0], 'MarkerEdgeColor',[0, 0, 0])
    if i == 1
        scatter(quad_coord(1, 11), quad_coord(2, 11), 65, 'MarkerFaceColor',[1, 0, 0])
    elseif i==2
        scatter(quad_coord(1, 4), quad_coord(2, 4),   65, 'MarkerFaceColor',[1, 0, 0])
        scatter(quad_coord(1, 11), quad_coord(2, 11), 65, 'MarkerFaceColor',[1, 0, 0])
        scatter(quad_coord(1, 18), quad_coord(2, 18), 65, 'MarkerFaceColor',[1, 0, 0])
    elseif i==3
        scatter(quad_coord(1, :), quad_coord(2, :), 65, 'MarkerFaceColor',[1, 0, 0])
    end 

    xlim([-gl/2-0.8, gl/2+0.8])
    axis equal; 
    set(gca,'XTickLabel',[]);
    set(gca,'YTickLabel',[]);
    set(gca,'xtick',[])
    set(gca,'ytick',[])
end 

set(gcf,'position',[5,5,400,300]);
saveas(gcf,'DAS_Cable_Point.png')


















