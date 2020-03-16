function dev = critfun(X,Y)
%% this function returns the deviance criterion of the Logistic regresion model
model = fitglm(X,Y,'Distribution','binomial');
dev = model.Deviance;
end