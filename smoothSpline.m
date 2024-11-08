function smooth_data = smoothSpline(x,y,p)
% Computes cubic spline smoothed array from noisy data
% x = Nx1 array of independent variable values
% y = Nx1 array of noisy data, y = f(x)
% p = smoothing parameter, 0 ≤ p ≤ 1
% p = 0 gives simple least-squares linear fit -- oversmooths all but
%       linear functions
% p = 1 is essentially no smoothing at all -- undersmooths

if ~exist('p','var')
    p = 0.1; 
end

pp = csaps(x,y,p);

smooth_data = fnval(pp,x);

end
