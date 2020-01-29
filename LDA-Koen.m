% W       = discovered linear coefficients (first column is the constants)
% X   = predictor data (variables in columns, observations in rows)
% Y  = target variable (class labels)
% Z  = vector of prior probabilities (optional)


function W = LDA(X,Y,Z)

% Determine size of input data
[n m] = size(X);

% Discover and count unique class labels
ClassLabel = unique(Y);
k = length(ClassLabel);

% Initialize
nGroup     = NaN(k,1);     % Group counts
GroupMean  = NaN(k,m);     % Group sample means
PooledCov  = zeros(m,m);   % Pooled covariance
W          = NaN(k,m+1);   % model coefficients

if  (nargin >= 3)  PriorProb = Z;  end

% Loop over classes to perform intermediate calculations
for i = 1:k,
    % Establish location and size of each class
    Group      = (Y == ClassLabel(i));
    nGroup(i)  = sum(double(Group));
    
    % Calculate group mean vectors
    GroupMean(i,:) = mean(X(Group,:));
    
    % Accumulate pooled covariance information
    PooledCov = PooledCov + ((nGroup(i) - 1) / (n - k) ).* cov(X(Group,:));
end

% Assign prior probabilities
if  (nargin >= 3)
    % Use the user-supplied priors
    PriorProb = Z;
else
    % Use the sample probabilities
    PriorProb = nGroup / n;
end

% Loop over classes to calculate linear discriminant coefficients
for i = 1:k,
    % Intermediate calculation for efficiency
    Temp = GroupMean(i,:) / PooledCov;
    
    % Constant
    W(i,1) = -0.5 * Temp * GroupMean(i,:)' + log(PriorProb(i));
    
    % Linear
    W(i,2:end) = Temp;
end
clear Temp

end 