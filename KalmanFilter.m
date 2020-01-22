function [ xi,P,predictedxi,predictedP ] = KalmanFilter( parameter_vector,y )
%%
% This function runs the Kalman filter for the scalar AR(1) model plus
% noise with a diffuse prior (roughly)

% Extract lenght of the data
T = numel(y);

% Extract the stuff we need from the input arguments
F = parameter_vector(1,1);
Q = parameter_vector(2,1);
R = parameter_vector(3,1);

xi_00 = 0;
P_00 = 10^6;

% The Kalman filter for AR(1)+noise
for t=1:T
    % Diffuse initialisation
    if t==1
    % Prediction for t=1
    predictedxi(t)  = F * xi_00;
    predictedP(t)   = F * P_00 * F' + Q;
    % Prediction for any other t
    else
    predictedxi(t)  = F * xi(t-1);
    predictedP(t)   = F * P(t-1) * F' + Q;
    end
    % Update for any t
    xi(t)  = predictedxi(t)  + predictedP(t) * 1/( predictedP(t) + R ) * ( y(t) - predictedxi(t) );
    P(t)   = predictedP(t)   - predictedP(t) * 1/( predictedP(t) + R ) * predictedP(t);
    % Close the loop over time
end
% Close the function
end

