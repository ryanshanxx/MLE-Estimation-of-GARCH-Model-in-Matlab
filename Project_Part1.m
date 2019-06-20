% initializing GARCH parameters
alpha0 = 0.0002;
alpha1=0.0679;
beta=0.9300;

% 10,000 sample paths with 1000 observations each
N=500;
n=1000;
% initializing PV, CI vectors and theta0
PV_alpha0=zeros(n,1);
PV_alpha1=zeros(n,1);
PV_beta=zeros(n,1);
CI_alpha0=zeros(n,2);
CI_alpha1=zeros(n,2);
CI_beta=zeros(n,2);
theta0=[alpha0,alpha1,beta];


for i=1:n
    % simulate GARCH(1,1) data
    Mdl = garch('Constant',alpha0,'ARCH',alpha1,'GARCH',beta);
    [var,Y] = simulate(Mdl,N);
    
    % Setting up fmincon
    A = [0 1 1]; 
    b = 1;
    Aeq = [];
    Ceq = [];
    lb = [0 0 0];
    ub = [];
    options = optimoptions('fmincon','Display','off');
    [theta,fval,exitflag,output,lambda,grid,hessian] = fmincon(@(theta)loglikeli(theta,Y),theta0, A, b, Aeq, Ceq, lb, ub,[],options);
    
    % calculate PV and CI
    sigma_hat=sqrt(abs(inv(hessian/N))/N);
    PV_alpha0(i)=2*(1-normcdf(abs(theta(1)-alpha0)/sigma_hat(1,1),0,1));
    PV_alpha1(i)=2*(1-normcdf(abs(theta(2)-alpha1)/sigma_hat(2,2),0,1));
    PV_beta(i)=2*(1-normcdf(abs(theta(3)-beta)/sigma_hat(3,3),0,1));
    z=norminv(1-(1-0.95)/2);
    CI_alpha0(i,1)=theta(1)+z*sigma_hat(1,1);
    CI_alpha0(i,2)=theta(1)-z*sigma_hat(1,1);
    CI_alpha1(i,1)=theta(2)+z*sigma_hat(2,2);
    CI_alpha1(i,2)=theta(2)-z*sigma_hat(2,2);
    CI_beta(i,1)=theta(3)+z*sigma_hat(3,3);
    CI_beta(i,2)=theta(3)-z*sigma_hat(3,3);
end 

reject_alpha0=0;
reject_alpha1=0;
reject_beta=0;
for i=1:n
    if PV_alpha0(i)<0.05
        reject_alpha0 = reject_alpha0+1;
    end
    if PV_alpha1(i)<0.05
        reject_alpha1 = reject_alpha1+1;
    end
    if PV_beta(i)<0.05
        reject_beta = reject_beta+1;
    end
end 

% calculate percentage of rejection
rej_per_alpha0=reject_alpha0/n;
rej_per_alpha1=reject_alpha1/n;
rej_per_beta=reject_beta/n;
 
function f = loglikeli(theta,Y)
n=size(Y(:,1),1);
sigma=zeros(n,1);
sigma(1)= sqrt(sum(Y.^2)/n);
f = log(1/(sqrt(2*pi)*sigma(1)))-Y(1)^2/(2*sigma(1)^2);
for i=2:n
    sigma(i)=sqrt(theta(1) + theta(2)*Y(i-1)^2+theta(3)*sigma(i-1)^2);
    l=log(1/(sqrt(2*pi)*sigma(i)))-Y(i)^2/(2*sigma(i)^2);
    f=f+l;
end
f=-f;
end 
