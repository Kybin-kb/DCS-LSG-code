function [B_trn,B_tst]=DCSLSG(exp_data,param)
which train;
WtrueTestTraining = exp_data.WTT ;
pos               = param.pos;
Xs                =    exp_data.Xs;
Xt                =    exp_data.train;
Ys                =    exp_data.ys;
Yt                =    exp_data.yt;
test              =    exp_data.test;

%% set parameters
setting.record = 0; %
setting.mxitr  = 8;
setting.xtol = 1e-5;
setting.gtol = 1e-5;
setting.ftol = 1e-8;

param.a1= 10; % MMD
param.a2= 10; % LSG of source domain
param.a3= 10; % LSG of target domain
param.a4= 1000; % Projection approximation
param.a5= 1; % Hash code learning
param.a6= 0.01; % The regularization term of hash function
param.a7= 0.1; % B



paras.max_iter   = 15;
dim = 16;
acc_ite= [];

[paras.nt,paras.d] =   size(Xt);
[paras.ns,paras.d] =   size(Xs);
%% leraning

X=[Xs;Xt];
X =  X';   % d*(ns+nt)
Xs =  Xs'; % d*ns
Xt =  Xt';
Xt0 = Xt;

[m,n] = size(X);
ns = size(Xs,2);
nt = size(Xt,2);
C = length(unique(Ys));
[vec,val]  =   eig(X*X');
[~,Idx]      =   sort(diag(val),'descend');
PCA          =   vec(:,Idx(1:C));
clear Idx;clear vec; clear val;



W = randn(C, param.r);
Bs  = sign(W'*PCA'*Xs);
Bt  = sign(W'*PCA'*Xt);

G = eye(paras.d);
P1=PCA;
P2=PCA;
I = eye(paras.d,paras.d);
I1 = eye(param.r,param.r);
I2 = eye(m,m);

Yt0 = [];
svmmodel = train(double(Ys), sparse(double(Xs')),'-s 1 -B 1.0 -q');
[Yt0,~,~] = predict(double(Yt), sparse(double(Xt')), svmmodel,'-q');


YsM              =     sparse(1:length(Ys), double(Ys), 1);
YsM             =     full(YsM);
YtM              =     sparse(1:length(Yt0), double(Yt0), 1);
YtM             =     full(YtM);
YsM = YsM';
YtM = YtM';
Xs2=Xs*Xs';
YXs=Xs*YsM';

for iter=1:paras.max_iter
    X0=[Xs,Xt0];
    nt0 = size(Xt0,2);
    
    % Estimate mu
    mu = estimate_mu(Xs',Ys,Xt0',Yt0);
    % Construct MMD matrix
    e = [1/ns*ones(ns,1);-1/nt0*ones(nt0,1)];
    M = e * e' * length(unique(Ys));
    N = 0;
    for c = reshape(unique(Ys),1,length(unique(Ys)))
        cYs=find(Ys==c);
        cYt=find(Yt0==c);
        if isempty(cYt)
            continue;
        end;
        cY=[cYs;ns+cYt];
        nc=length(cY);
        nc_Xs=length(cYs);
        nc_Xt=length(cYt);
        e=zeros(nc,1);
        e(1:nc_Xs)=1/nc_Xs;
        e(nc_Xs+1:end)=-1/nc_Xt;
        N(cY,cY)=e*e';
    end
    M = (1 - mu) * M + mu * N;
    M = M / norm(M,'fro');
    K= X0*M*X0';
    Bs = sign(inv(W'*W+param.a7*I1)*W'*P1'*Xs);
    Bt = sign(inv(W'*W+param.a7*I1)*W'*P2'*Xt0);
    XBs = Xs*Bs';
    XBt = Xt0*Bt';
    W = (P1'*XBs+P2'*Xt0*Bt')*inv(Bs*Bs'+Bt*Bt');
    P1 = inv(2*param.a1*K+(2+2*param.a2)*Xs2+(2*param.a4)*I)*(2*param.a2*YXs+2*param.a4*P2+2*XBs*W'-2*param.a1*K*P2);
    P2 = inv(2*param.a1*K+(2+2*param.a3)*Xt0*Xt0'+(2*param.a4)*I)*(2*param.a3*Xt0*YtM'+2*param.a4*P1+2*XBt*W'-2*param.a1*K*P1);
    
    [Yt0,predLabels,Xt0] = seletPseudoLabels(Xs,Xt,Ys,C,iter,paras.max_iter,P1,P2);
    Xt0 = Xt0';
    Yt0 = Yt0';
    
    Zs = Xs'*P1;
    Zt = Xt'*P2;
    
    svmmodel = train(double(Ys), sparse(double(Zs)),'-s 1 -B 1.0 -q');
    [Yt1,~,~] = predict(double(Yt), sparse(double(Zt)), svmmodel,'-q');
    acc=getAcc(Yt1,Yt);
    acc_ite=[acc_ite,acc];
    
    YtM              =     sparse(1:length(Yt0), double(Yt0), 1);
    YtM             =     full(YtM);
    YtM = YtM';
    
    r3 = size(YtM,1);
    if r3 ~= C
        for r1=1:C
            flag = find(Yt0==r1);
            if flag
                continue;
            else
                r2 = size(YtM,2);
                B3=zeros(1,r2);
                B1=YtM(1:r1-1,:);
                if r1 == C
                    YtM = [B1;B3];
                else
                    B2=YtM(r1:C-1,:);
                    YtM = [[B1;B3];B2];
                end;
            end;
        end;
    end;
    
    
    B = [Bs,Bt];
    H = B*X0'*inv(X0*X0'+param.a6*I2);
    
    if strcmp(param.retrieval, 'cross-domain')
        B_train           =    (Xs'*H'>0);
    elseif strcmp(param.retrieval, 'single-domain')
        B_train           =    (Xt'*H'>0);
    end
    
    B_test            =    (test*H'>0);
    B_trn             =    compactbit(B_train);
    B_tst             =    compactbit(B_test);
end
