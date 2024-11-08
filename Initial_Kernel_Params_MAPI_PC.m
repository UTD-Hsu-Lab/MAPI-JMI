clear, close all; clc;

%% Importing Data

df_data = readtable('240507_MAPI_PC_Conditions_LHS25.xlsx');
df_data = table2array(df_data);
data = df_data(:, [2, 3, 5, 6, 12]); % Change the final column to vary the output metric.

% Input Data
x1 = data(:,3);
x2 = data(:,1);
x3 = data(:,2);
x4 = data(:,4);
x_data = [x1 x2 x3 x4];

% Output Data
% Procrustes/Frechet Output
y = abs(log(data(:,5)/(data(8,5))));

% 1 - Cosine Similarity Output
% y = 1 - data(:,5)/data(8,5);

% MSE Output
% y = 1./(data(:,5));

%% Defining input variable names, setting min/max, and normalizing the variables

% Define pulse length range
x1min = 10;
x1max = 50;

% Define number of micropulses range
x2min = 2;
x2max = 30;

% Define duty cycle range
x3min = 20;
x3max = 70;

% Define radiant energy range
x4min = 3;
x4max = 13.5;

% Normalize Data
x1scale = (x1 - x1min)/(x1max - x1min);
x2scale = (x2 - x2min)/(x2max - x2min);
x3scale = (x3 - x3min)/(x3max - x3min);
x4scale = (x4 - x4min)/(x4max - x4min);

%% Length Scale for Pulse Length
figure1 = figure;
f1 = fit(x1scale,y,'smoothingspline');
x1_plot = plot(f1,x1scale,y);
xlabel('Scaled Pulse Length')
ylabel('Scaled Procrustes Distance')
title('Pulse Length vs Procrustes Distance')
legend('Raw Data','SmoothSpline','Location','northwest')

saveas(figure1,'Length Scale for Pulse Length.png')

% Evalulate the fit curve in the interval of [0,1]
xeval = linspace(0,1,1001)';
y1_eval = feval(f1,xeval);
localmax_x1 = islocalmax(y1_eval);
localmax_x1 = localmax_x1.*y1_eval;
localmax_x1 = [xeval,localmax_x1];

% Find rows where either of the columns is zero
rows_to_remove = any(localmax_x1 == 0, 2);

% Remove rows with a zero in one of the two columns
localmax_x1 = localmax_x1(~rows_to_remove, :);
localmax_x1 = localmax_x1(:,1);

% Initialize an empty matrix to store the results
x1_p2p = zeros(size(localmax_x1));

% Subtract the last value from the next value
for i = 2:numel(localmax_x1)
    x1_p2p(i) = localmax_x1(i) - localmax_x1(i-1);
end

x1_p2p = abs(x1_p2p(2:end,:));
avgx1_p2p = sum(x1_p2p)/numel(x1_p2p);
x1lengthscale = avgx1_p2p;

%% Length Scale for # of Micropulses

figure2 = figure;
f2 = fit(x2scale,y,'smoothingspline');
plot(f2,x2scale,y)
xlabel('Scaled # of \mupulses')
ylabel('Scaled Procrustes Distance')
title('# of \mupulses vs Procrustes Distance')
legend('Raw Data','SmoothSpline','Location','northwest')
saveas(figure2,'Length Scale for Micropulses.png')

% Evalulate the fit curve in the interval of [0,1]
y2_eval = feval(f2,xeval);
localmax_x2 = islocalmax(y2_eval);
localmax_x2 = localmax_x2.*y2_eval;
localmax_x2 = [xeval,localmax_x2];

% Find rows where either of the columns is zero
rows_to_remove = any(localmax_x2 == 0, 2);

% Remove rows with a zero in one of the two columns
localmax_x2 = localmax_x2(~rows_to_remove, :);
localmax_x2 = localmax_x2(:,1);
 
% Initialize an empty matrix to store the results
x2_p2p = zeros(size(localmax_x2));

% Subtract the last value from the next value
for i = 2:numel(localmax_x2)
    x2_p2p(i) = localmax_x2(i) - localmax_x2(i-1);
end

x2_p2p = abs(x2_p2p(2:end,:));
avgx2_p2p = sum(x2_p2p)/numel(x2_p2p);
x2lengthscale = avgx2_p2p;

%% Length Scale for Duty Cycle
figure3 = figure;
f3 = fit(x3scale,y,'smoothingspline');
plot(f3,x3scale,y)
xlabel('Scaled Duty Cycle')
ylabel('Scaled Procrustes Distance')
title('Duty Cycle vs Procrustes Distance')
legend('Raw Data','SmoothSpline','Location','northwest')
saveas(figure3,'Length Scale for Duty Cylce.png')

% Evalulate the fit curve in the interval of [0,1]
y3_eval = feval(f3,xeval);
localmax_x3 = islocalmax(y3_eval);
localmax_x3 = localmax_x3.*y3_eval;
localmax_x3 = [xeval,localmax_x3];

% Find rows where either of the columns is zero
rows_to_remove = any(localmax_x3 == 0, 2);

% Remove rows with a zero in one of the two columns
localmax_x3 = localmax_x3(~rows_to_remove, :);
localmax_x3 = localmax_x3(:,1);

% Initialize an empty matrix to store the results
x3_p2p = zeros(size(localmax_x3));

% Subtract the last value from the next value
for i = 2:numel(localmax_x3)
    x3_p2p(i) = localmax_x3(i) - localmax_x3(i-1);
end

x3_p2p = abs(x3_p2p(2:end,:));
avgx3_p2p = sum(x3_p2p)/numel(x3_p2p);
x3lengthscale = avgx3_p2p;

%% Length Scale for Radiant Energy
figure4 = figure;
scatter(x4scale,y)
f4 = fit(x4scale,y,'smoothingspline');
plot(f4,x4scale,y)
xlabel('Scaled Radiant Energy')
ylabel('Scaled Procrustes Distance')
title('Radiant Energy vs Procrustes Distance')
legend('Raw Data','SmoothSpline','Location','northwest')
saveas(figure4,'Length Scale for Radiant Energy.png')

% Evalulate the fit curve in the interval of [0,1]
y4_eval = feval(f4,xeval);
localmax_x4 = islocalmax(y4_eval);
localmax_x4 = localmax_x4.*y4_eval;
localmax_x4 = [xeval,localmax_x4];

% Find rows where either of the columns is zero
rows_to_remove = any(localmax_x4 == 0, 2);

% Remove rows with a zero in one of the two columns
localmax_x4 = localmax_x4(~rows_to_remove, :);
localmax_x4 = localmax_x4(:,1);

% Initialize an empty matrix to store the results
x4_p2p = zeros(size(localmax_x4));

% Subtract the last value from the next value
for i = 2:numel(localmax_x4)
    x4_p2p(i) = localmax_x4(i) - localmax_x4(i-1);
end

x4_p2p = abs(x4_p2p(2:end,:));
avgx4_p2p = sum(x4_p2p)/numel(x4_p2p);
x4lengthscale = avgx4_p2p;

lengthscale = [x1lengthscale; x2lengthscale; x3lengthscale; x4lengthscale];

%% Calculating Scale Factor

maxy = max(y);
miny = min(y);

scalefactor = (maxy - miny)/2;

initialkparams = [lengthscale; scalefactor];
