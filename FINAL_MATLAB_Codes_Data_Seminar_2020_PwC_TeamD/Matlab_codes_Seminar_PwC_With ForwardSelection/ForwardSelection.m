function [mincoefs]= ForwardSelection(X,y)
%% Do forward selection with logistic regression, with given input X and y. (of a given quarter of  year)
model0 = fitglm(X,y,'Distribution','binomial');
dev0 = model0.Deviance;

maxdev = chi2inv(.95,1);     
opt = statset('display','iter','TolFun',maxdev,'TolTypeFun','abs');

inmodel = sequentialfs(@critfun,X,y,'cv','none','nullmodel',true,'options',opt,'direction','forward');

mincoefs=find(inmodel==1);

end
