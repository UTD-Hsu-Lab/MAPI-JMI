clear, close all; clc;

%% Importing Data

df_data = readtable('240507_MAPI_PC_Conditions_LHS25.xlsx');
df_data = table2array(df_data);
data = df_data(:, [2, 3, 5, 6, 10]); % Change the final column to vary the output metric.

% Input Data
x1 = data(:,3);
x2 = data(:,1);
x3 = data(:,2);
x4 = data(:,4);
x_data = [x1 x2 x3 x4];

% Output Data
% Procrustes/Frechet Output/RMSE?
y_data = abs(log(data(:,5)/(data(8,5))));

% 1 - Cosine Similarity Output
% y = 1 - data(:,5)/data(8,5);

% MSE/RMSE Output
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

xmin = [x1min, x2min, x3min, x4min];
xmax = [x1max, x2max, x3max, x4max];

% Normalize the data
x_scaled = zeros(size(x_data));

for i = 1:4
    x_scaled(:,i) = (x_data(:,i) - xmin(i))/(xmax(i) - xmin(i));
end

% Setting the Domain (Normalized)
Xdomain = [0 1 ; 0 1; 0 1; 0 1];

%% Select kernel type (uncomment one of the following lines)
%kerneltype = 'ardsquaredexponential';
kerneltype = 'ardmatern52';
%kerneltype = 'ardmatern32';

sigma0 = 0.309; % Initial data uncertainty, sigma0 must be > 0
% Procrustes 0.846
% Procrustes w/o 0.869
% Frechet 0.309 inflection at ~0.193
% RMSE 1.10

%% Set up 5D hypercube grid
x1 = linspace(0,1,41);
x2 = linspace(0,1,29);
x3 = linspace(0,1,11);
x4 = linspace(0,1,106);


[X1,X2,X3,X4] = ndgrid(x1,x2,x3,x4);

X = [X1(:), X2(:), X3(:), X4(:)];

%% Gaussian process regression model using fitrgp
kparams0 = [0.088, 0.284, 0.154, 0.087, 1.88];
% Procrustes [0.117, 0.384, 0.329, 0.107, 3.16]
% Procrustes w/o [0.090, 0.378, 0.379, 0.097, 4.07]
% Frechet [0.088, 0.284, 0.154, 0.087, 1.88]
% Frechet changed [0.1, 0.284, 0.154, 0.1, 1.88]
% RMSE [0.117, 0.260, 0.400, 0.075, 2.78]
mdl = fitrgp(x_scaled,y_data,...
        KernelFunction=kerneltype,...
        Sigma=sigma0,ConstantSigma=true,KernelParameters=kparams0,...
        FitMethod='none');
    
[y_pred_n,sd] = predict(mdl,X);  % y_pred_n calculated using ndgrid coords
kernparam = mdl.KernelInformation.KernelParameters; % Extract optimized kernel parameters

%% Acquisition Function: Upper Confidence Bound policy
AFtype = 'UCB';
UCB_weight = 1; % Exploration-exploitation parameter
                    % Higher weight (e.g., 10) = more exploration
                    % Lower weight (e.g., 1) = more exploitation                    
AF = y_pred_n + sd * UCB_weight;  % Maximization

%% Next Batch Selection

% Maximization (Single Point)
[next_y, idx] = max(y_pred_n);
next_y = exp(-next_y)*data(8,5);
xnext_scaled = X(idx, :);
xnext = zeros(1, 4);

for i = 1:4
    xnext(i) = (xmax(i) - xmin(i)) * xnext_scaled(i) + xmin(i);
end

fprintf('The minimized output is: %.4f\n', next_y);
fprintf('The corresponding x-values for the minimized output are: [%.1f %.1f %.1f %.1f]\n', xnext);

% Local Penalization
% batchsize = 4; % The number of output batches
% next_batchLP = LocalPenal_4D_MAPI(AF,sd,y_pred_n,X,batchsize);
% 
% nbLPx1 = next_batchLP(1:batchsize,1);
% nbLPx2 = next_batchLP(1:batchsize,2);
% nbLPx3 = next_batchLP(1:batchsize,3);
% nbLPx4 = next_batchLP(1:batchsize,4);
% 
% x1nextLP = (x1max - x1min)*nbLPx1 + x1min;
% x2nextLP = (x2max - x2min)*nbLPx2 + x2min;
% x3nextLP = (x3max - x3min)*nbLPx3 + x3min;
% x4nextLP = (x4max - x4min)*nbLPx4 + x4min;
% next_batchLP = [x1nextLP x2nextLP x3nextLP x4nextLP];

%% Max Projection Slice Plots 4D into 2D

x1 = (x1max - x1min)*x1 + x1min;
x2 = (x2max - x2min)*x2 + x2min;
x3 = (x3max - x3min)*x3 + x3min;
x4 = (x4max - x4min)*x4 + x4min;

Ycube = reshape(y_pred_n,[41,29,11,106]);

% Find max value along x3 & x4 coordinates for every point in (x1,x2)-plane
M12 = max(Ycube,[],[3 4]); M12 = squeeze(M12);

% Find max value along x2 & x4 coordinate for every point in (x3,x1)-plane
M13 = max(Ycube,[],[2 4]); M13 = squeeze(M13);

% Find max value along x1 & x4 coordinate for every point in (x2,x3)-plane
M23 = max(Ycube,[],[1 4]); M23 = squeeze(M23);

% Find max value along x2 & x3 coordinate for every point in (x1,x4)-plane
M14 = max(Ycube,[],[2 3]); M14 = squeeze(M14);

% Find max value along x1 & x3 coordinate for every point in (x2,x4)-plane
M24 = max(Ycube,[],[1 3]); M24 = squeeze(M24);

% Find max value along x1 & x2 coordinate for every point in (x3,x4)-plane
M34 = max(Ycube,[],[1 2]); M34 = squeeze(M34);

fig1 = figure(1); 
% Modify the position [left, bottom, width, height]
fig1.Position = [10 10 1500 1000]; % for (2x3) subplots
% fig1.Position = [10 10 600 1500]; % for (3x2) subplots
rows_to_include = [23:25,27];
last_batch = x_data(rows_to_include,:);
% subplot(n,m,p)
subplot(231);
    % pos1 = get(gca, 'Position');
    % set(gca, 'Position', [pos1(1) pos1(2) pos1(3)*(5/6) pos1(4)]); % figure position for (3x2) subplots
    [X_1,X_2]=ndgrid(x1,x2);
    contourf(X_1,X_2,M12,50,'FaceColor','flat','LineStyle','none');
    xlabel('Pulse Length','FontSize',15); ylabel('# of \mupulses','FontSize',15);
    % cb = colorbar; colormap parula;
    % title('Projection of max(x3 & 4) onto x1,x2 plane');
hold on;
    scatter(x_data(1:19,1),x_data(1:19,2),40,'green','filled','MarkerEdgeColor','k',...
        'LineWidth',1.0);
    scatter(x_data(20:22,1),x_data(20:22,2),40,'blue','filled','MarkerEdgeColor','k',...
        'LineWidth',1.0);
    scatter(last_batch(:,1),last_batch(:,2),40,'red','filled','MarkerEdgeColor','k',...
        'LineWidth',1.0);
        scatter(x_data(26,1),x_data(26,2),150,'red','filled','Marker','pentagram','MarkerEdgeColor','k',...
        'LineWidth',1.0);
    scatter(x_data(28:31,1),x_data(28:31,2),40,'cyan','filled','MarkerEdgeColor','k',...
        'LineWidth',1.0);
    % scatter(x1nextLP,x2nextLP,20,'black','filled','MarkerEdgeColor','k',...
    %     'LineWidth',1.0);
hold off;
%
subplot(232);
    % pos1 = get(gca, 'Position');
    % set(gca, 'Position', [pos1(1) pos1(2) pos1(3)*(5/6) pos1(4)]); % figure position for (3x2) subplots
    [X_1,X_3]=ndgrid(x1,x3);
    contourf(X_1,X_3,M13,50,'FaceColor','flat','LineStyle','none');
    xlabel('Pulse Length','FontSize',15); ylabel('Duty Cycle','FontSize',15);
    % cb = colorbar; colormap parula;
    % title('Projection of max(x2 & x4) onto x3,x1 plane');
hold on;
    scatter(x_data(1:19,1),x_data(1:19,3),40,'green','filled','MarkerEdgeColor','k',...
        'LineWidth',1.0);
    scatter(x_data(20:22,1),x_data(20:22,3),40,'blue','filled','MarkerEdgeColor','k',...
        'LineWidth',1.0);
    scatter(last_batch(:,1),last_batch(:,3),40,'red','filled','MarkerEdgeColor','k',...
        'LineWidth',1.0);
    scatter(x_data(26,1),x_data(26,3),150,'red','filled','Marker','pentagram','MarkerEdgeColor','k',...
        'LineWidth',1.0);
    scatter(x_data(28:31,1),x_data(28:31,3),40,'cyan','filled','MarkerEdgeColor','k',...
        'LineWidth',1.0);
    % scatter(x1nextLP,x3nextLP,20,'black','filled','MarkerEdgeColor','k',...
    %     'LineWidth',1.0);
hold off;
%
subplot(233);
    % pos1 = get(gca, 'Position');
    % set(gca, 'Position', [pos1(1) pos1(2) pos1(3)*(5/6) pos1(4)]); % figure position for (3x2) subplots
    [X_2,X_3]=ndgrid(x2,x3);
    contourf(X_2,X_3,M23,50,'FaceColor','flat','LineStyle','none');
    xlabel('# of \mupulses','FontSize',15); ylabel('Duty Cycle','FontSize',15);
    cb = colorbar; colormap parula;
    cbPos = get(cb, 'Position');
    set(cb, 'Position', [cbPos(1)+0.05, cbPos(2)-0.325,cbPos(3)*1.25,cbPos(4)*1.25]);
    % title('Projection of max(x1 & x4) onto x2,x3 plane');
hold on;
    scatter(x_data(1:19,2),x_data(1:19,3),40,'green','filled','MarkerEdgeColor','k',...
        'LineWidth',1.0);
    scatter(x_data(20:22,2),x_data(20:22,3),40,'blue','filled','MarkerEdgeColor','k',...
        'LineWidth',1.0);
    scatter(last_batch(:,2),last_batch(:,3),40,'red','filled','MarkerEdgeColor','k',...
        'LineWidth',1.0);
    scatter(x_data(26,2),x_data(26,3),150,'red','filled','Marker','pentagram','MarkerEdgeColor','k',...
        'LineWidth',1.0);
    scatter(x_data(28:31,2),x_data(28:31,3),40,'cyan','filled','MarkerEdgeColor','k',...
        'LineWidth',1.0);
    % scatter(x2nextLP,x3nextLP,20,'black','filled','MarkerEdgeColor','k',...
    %     'LineWidth',1.0);
hold off;
%
subplot(234);
    % pos1 = get(gca, 'Position');
    % set(gca, 'Position', [pos1(1) pos1(2) pos1(3)*(5/6) pos1(4)]); % figure position for (3x2) subplots
    [X_1,X_4]=ndgrid(x1,x4);
    contourf(X_1,X_4,M14,50,'FaceColor','flat','LineStyle','none');
    xlabel('Pulse Length','FontSize',15); ylabel('Radiant Energy','FontSize',15);
    % cb = colorbar; colormap parula;
    % cbPos = get(cb, 'Position');
    % set(cb, 'Position', [cbPos(1)+0.1, cbPos(2)*(1/1.25),cbPos(3)*1.25,cbPos(4)*1.25]); % color bar for (3x2) subplots
    % title('Projection of max(x2 & x3) onto x1,x4 plane');
hold on;
    scatter(x_data(1:19,1),x_data(1:19,4),40,'green','filled','MarkerEdgeColor','k',...
        'LineWidth',1.0);
    scatter(x_data(20:22,1),x_data(20:22,4),40,'blue','filled','MarkerEdgeColor','k',...
        'LineWidth',1.0);
    scatter(last_batch(:,1),last_batch(:,4),40,'red','filled','MarkerEdgeColor','k',...
        'LineWidth',1.0);
    scatter(x_data(26,1),x_data(26,4),150,'red','filled','Marker','pentagram','MarkerEdgeColor','k',...
        'LineWidth',1.0);
    scatter(x_data(28:31,1),x_data(28:31,4),40,'cyan','filled','MarkerEdgeColor','k',...
        'LineWidth',1.0);
    % scatter(x1nextLP,x4nextLP,20,'black','filled','MarkerEdgeColor','k',...
    %     'LineWidth',1.0);
hold off;
%
subplot(235);
    % pos1 = get(gca, 'Position');
    % set(gca, 'Position', [pos1(1) pos1(2) pos1(3)*(5/6) pos1(4)]); % figure position for (3x2) subplots
    [X_2,X_4]=ndgrid(x2,x4);
    contourf(X_2,X_4,M24,50,'FaceColor','flat','LineStyle','none');
    xlabel('# of \mupulses','FontSize',15); ylabel('Radiant Energy','FontSize',15);
    % cb = colorbar; colormap parula;
    % title('Projection of max(x1 & x3) onto x2,x4 plane');
hold on;
    scatter(x_data(1:19,2),x_data(1:19,4),40,'green','filled','MarkerEdgeColor','k',...
        'LineWidth',1.0);
    scatter(x_data(20:22,2),x_data(20:22,4),40,'blue','filled','MarkerEdgeColor','k',...
        'LineWidth',1.0);
    scatter(last_batch(:,2),last_batch(:,4),40,'red','filled','MarkerEdgeColor','k',...
        'LineWidth',1.0);
    scatter(x_data(26,2),x_data(26,4),150,'red','filled','Marker','pentagram','MarkerEdgeColor','k',...
        'LineWidth',1.0);
    scatter(x_data(28:31,2),x_data(28:31,4),40,'cyan','filled','MarkerEdgeColor','k',...
        'LineWidth',1.0);
    % scatter(x2nextLP,x4nextLP,20,'black','filled','MarkerEdgeColor','k',...
    %     'LineWidth',1.0);
hold off;
%
subplot(236);
    % pos1 = get(gca, 'Position');
    % set(gca, 'Position', [pos1(1) pos1(2) pos1(3)*(5/6) pos1(4)]); % figure position for (3x2) subplots  
    [X_3,X_4]=ndgrid(x3,x4);
    contourf(X_3,X_4,M34,50,'FaceColor','flat','LineStyle','none');
    xlabel('Duty Cycle','FontSize',15); ylabel('Radiant Energy','FontSize',15);
    % cb = colorbar; colormap parula;
    % cbPos = get(cb, 'Position');
    % set(cb, 'Position', [cbPos(1)+0.09, cbPos(2)+0.18, cbPos(3)*2, cbPos(4)*2]);
    % title('Projection of max(x1 & x2) onto x3,x4 plane');
hold on;
    scatter(x_data(1:19,3),x_data(1:19,4),40,'green','filled','MarkerEdgeColor','k',...
        'LineWidth',1.0);
    scatter(x_data(20:22,3),x_data(20:22,4),40,'blue','filled','MarkerEdgeColor','k',...
        'LineWidth',1.0);
    scatter(last_batch(:,3),last_batch(:,4),40,'red','filled','MarkerEdgeColor','k',...
        'LineWidth',1.0);
    scatter(x_data(26,3),x_data(26,4),150,'red','filled','Marker','pentagram','MarkerEdgeColor','k',...
        'LineWidth',1.0);
    scatter(x_data(28:31,3),x_data(28:31,4),40,'cyan','filled','MarkerEdgeColor','k',...
        'LineWidth',1.0);
    % scatter(x3nextLP,x4nextLP,20,'black','filled','MarkerEdgeColor','k',...
    %     'LineWidth',1.0);
    % legend('','LHS Conditions (1-20)','Round #1 (21-23)','Round #2 (24-28)','Best Point (25)','Round #3 (29-31)')
hold off;
%

saveas(fig1,'MaxProj4Dto2D.png')

%% Parity Plotting for a Given Model

% Modeling the Predicted vs Experimental Data
y_model = predict(mdl,x_scaled); % Use the model to predict y values at each scaled x data point
lin_mdl = fitlm(y_data,y_model); % Create a linear fit between the experimental and predicted outputs
y_fit = predict(lin_mdl,y_data); % Use the linear fit to create a set of y variables 
RSquared = round(lin_mdl.Rsquared.Ordinary,3); % Grab the R^2 value from the fit parameters

% Get the sorting indices for the first vector
[~, sortIdx] = sort(y_fit);

% Reorder both vectors based on the sorted indices
sorted_y_fit = y_fit(sortIdx);
sorted_y_data = y_data(sortIdx);

% Grab the slope and y-intercept from the fit parameters
A = table2array(lin_mdl.Coefficients);
A1 = round(A(2,1),3);
A2 = round(A(1,1),3);

% Plot the Data and the Linear Best Fit Line
fig2 = figure;
scatter(y_data,y_model,'magenta','filled','MarkerEdgeColor','k',...
        'LineWidth',1.0);
hold on
plot(sorted_y_data,sorted_y_fit,'k--', 'LineWidth', 2)
xlabel('Experimental Procrustes Distance')
ylabel('Predicted Procrustes Distance')
set(gca, 'Box', 'on', 'LineWidth', 2);
% title('Parity Plot for GPR BO Model')
% legend('data','linear best fit','location','best')

formatSpec1 = "y = %gx + %g";
formatSpec2 = "r^{2} = %g";
str1 = sprintf(formatSpec1,A1,A2);
str2 = sprintf(formatSpec2,RSquared);
hold off

saveas(fig2,'ParityPlot.png')
