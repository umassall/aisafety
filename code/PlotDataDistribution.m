m = 50000;                  % The number of points to sample
X = normrnd(0,1,m,1);       % Sample standard normal m times
Y = X + normrnd(0,1,m,1);   % Set Y based on X
h = scatter(X,Y, 'filled'); % Plot the points
h.MarkerFaceAlpha = 0.01;   % Make markers transparent
set(gcf,'color','w');       % Set the background of the plot to white
xlabel('X');                % Label the x-axis
xlim([-3,3]);               % Set the range of the x-axis
xticks([-3,0,3]);           % Set where the x-axis tick marks are
ylabel('Y');                % Label the y-axis
ylim([-3,3]);               % Set the range of the y-axis
yticks([-3,0,3]);           % Set where the y-axis tick marks are
set(gca,'FontSize',18)      % Change the font size
export_fig('samples.png','-m2'); % Use the export_fig library for higher resolution: https://www.mathworks.com/matlabcentral/fileexchange/23629-export_fig