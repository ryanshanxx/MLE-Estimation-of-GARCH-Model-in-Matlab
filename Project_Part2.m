%Step1. load the data and plot the original price data and return data
% input data
PriceSeries = readtable("AAPL.csv");
prices = PriceSeries(:,6);
prices = table2array(prices);
t = (0:length(prices)-1);

% plot price
figure(1);
plot(t,prices);
ylabel('Price');
title('PriceSeries2');


% calculate log return
ret = zeros(1,length(prices));
for i = 1:(length(prices)-1)
    ret(i) = log(prices(i+1)/prices(i));
end
t1 = (0:length(ret)-1);

% plot return -- observe volatility clustering
figure(2);
plot(t1,ret);
ylabel('Return')
title('PriceSeries2 Return')


%Step2. check for corraltion in the return series
figure(3);
autocorr(ret);
title('ACF with Bounds for Raw Return Series'); % no autocorrelation in raw return series
figure(4);
parcorr(ret);
title('PCAF with Bounds for Raw Return Series');
figure(5);
autocorr(ret.^2);
title('ACF of the Squared Return') % autocorrelation in raw return series

%Step3. Pre-estimation step
[h1,pValue1,stat1,cValue1] = lbqtest(ret,'lags',[5,10,15],'Alpha',0.05);%Ljung-Box-Pierce Q-test
[h1' pValue1' stat1' cValue1'];
[h2,pValue2,stat2,cValue2] = lbqtest(ret.^2,'lags',[5,10,15],'Alpha',0.05);%Ljung-Box-Pierce Q-test for squared return
[h2' pValue2' stat2' cValue2'];
[h3,pValue3,stat3,cValue3] = archtest(ret.^2,'lags',[5,10,15],'Alpha',0.05);%Engle's ARCH Test
[h3' pValue3' stat3' cValue3'];  % comment see 16/17

%Step4. Parameter Estimation
% garch(1,1)
Mdl_11 = garch(1,1);
[estMdl_11,estParamCov_11,logL_11,residuals_11] = estimate(Mdl_11,ret.'); %GARCH(1,1) model
%summarize(estMdl_11);

%Step5. Post-estimation Analysis
%compare the reiduals, conditional std deviations and returns
condVol_11 = sqrt(infer(estMdl_11,ret.')); %conditional standard deviations garch(1,1)
figure(6);
plot(condVol_11);
title('Inferred Volatitlity Garch(1,1)');

%Plot and Compare correlation of std residual figure
stdResids_11 = ret./condVol_11.'; %compute conditional std deviations
squarestdResids_11 = stdResids_11.^2;

figure(7);
plot(stdResids_11);
title('Conditional std Deviations Garch(1,1)');

figure(8);
subplot(2,1,1);
autocorr(stdResids_11);
title('ACF with Returns Garch(1,1)')
subplot(2,1,2);
autocorr(squarestdResids_11);
title('ACF with Squared Returns Garch(1,1)')

%Quatify and Compare Correlation of the Standardized Innovations.
[h4,pValue4,stat4,cValue4] = lbqtest((stdResids_11./condVol_11.'),'lags',[5,10,15],'Alpha',0.05);
[h4' pValue4' stat4' cValue4'];
[h5,pValue5,stat5,cValue5] = lbqtest((stdResids_11./condVol_11.').^2,'lags',[5,10,15],'Alpha',0.05);
[h5' pValue5' stat5' cValue5'];

