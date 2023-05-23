function [PI, error] = segment(P, arc_len)
    
    % retrieve data
    Px = P(:, 1);
    Py = P(:, 2);
    Pz = P(:, 3);

    % get the approximated npts
    chordlen = sum(sqrt(sum(diff(P,[],1).^2,2)));
    npts = floor(chordlen / arc_len);
    
    % try many npts and find the one yeilds minimum error
    extra = 30;
    npts_try = npts - extra : npts + extra;
    n_try = length(npts_try);
    interval = zeros(n_try, 1);
    
    for ipts = 1 : n_try
        PI = interparc(npts_try(ipts), Px, Py, Pz, 'spline');
        Pxi = PI(:,1);
        Pyi = PI(:,2);
        Pzi = PI(:,3);
        interval(ipts) = mean(sqrt(diff(Pxi).^2 + diff(Pyi).^2 + diff(Pzi).^2));
    end
    
    % find the npts with minimum errors
    [~, min_index] = min(abs(interval - arc_len));
    PI = interparc(npts_try(min_index), Px, Py, Pz, 'spline');
    
    Pxi = PI(:,1);
    Pyi = PI(:,2);
    Pzi = PI(:,3);
    error = abs(mean(sqrt(diff(Pxi).^2 + diff(Pyi).^2 + diff(Pzi).^2)) - arc_len);

    fprintf("The mean error between desired arc length and compuated one is %.6f\n", error);

end 




