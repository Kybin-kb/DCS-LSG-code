function [exp_data]=construct_dataset(db_name,param,runtimes)
topNum=0;%PWCF
addpath('./DB/');

%%choose and load dataset
if strcmp(db_name,'VOC2007&Caltech101')
    load VOC2007 %source data
    Xs = double(data(:,1:end-1));
    Xs = normalize1(Xs);%Xs/max(max(abs(Xs)));
    ys = data(:,end);
    clear data; %clear labels;
    
    load Caltech101 %target data
    Xt = double(data(:,1:end-1));
    Xt = normalize1(Xt);%Xt/max(max(abs(Xt)));
    yt = data(:,end);
    clear data; %clear labels;
    
elseif strcmp(db_name, 'Caltech256&ImageNet')
    load dense_imagenet_decaf7_subsampled; %target data
    Xt = normalize1(fts);
    Xt = double(Xt);
    yt = double(labels);
    clear fts; clear labels;
    load dense_caltech256_decaf7_subsampled; %source data
    Xs = normalize1(fts);%fea;
    Xs = double(Xs);
    ys = double(labels);
    clear fts; clear labels;
    
elseif strcmp(db_name,'MNIST&USPS')
    load MNIST_vs_USPS.mat %source data
    Xs = double(X_src)';
    Xs = normalize1(Xs);%Xs/max(max(abs(Xs)));
    ys = double(Y_src); %vector label
    Xt = double(X_tar)';
    Xt = normalize1(Xt);%Xs/max(max(abs(Xs)));
    yt = double(Y_tar); %vector label
    clear X_src;  clear Y_src;  clear X_tar;  clear Y_tar;
    
elseif strcmp(db_name,'Product&Real')
    load Product_vgg16_fc7.mat %source data
    Xs = double(fts);
    Xs = normalize1(Xs);%Xs/max(max(abs(Xs)));
    ys = double(labels); %vector label
    clear fts; clear labels;
    
    load RealWorld_vgg16_fc7.mat %target data
    Xt = double(fts);
    Xt = normalize1(Xt);%Xs/max(max(abs(Xs)));
    yt = double(labels); %vector label
    clear fts; clear labels;
    
elseif strcmp(db_name,'Real&Product')
    load RealWorld_vgg16_fc7.mat %source data
    Xs = double(fts);
    Xs = normalize1(Xs);%Xs/max(max(abs(Xs)));
    ys = double(labels); %vector label
    clear fts; clear labels;
    
    load Product_vgg16_fc7.mat %target data
    Xt = double(fts);
    Xt = normalize1(Xt);%Xs/max(max(abs(Xs)));
    yt = double(labels); %vector label
    clear fts; clear labels;
    
elseif strcmp(db_name,'Clipart&Real')
    load Clipart_vgg16_fc7.mat %source data
    Xs = double(fts);
    Xs = normalize1(Xs);%Xs/max(max(abs(Xs)));
    ys = double(labels); %vector label
    clear fts; clear labels;
    load RealWorld_vgg16_fc7.mat %target data
    Xt = double(fts);
    Xt = normalize1(Xt);%Xs/max(max(abs(Xs)));
    yt = double(labels); %vector label
    clear fts; clear labels;
elseif strcmp(db_name,'Real&Clipart')
    load RealWorld_vgg16_fc7.mat %source data
    Xs = double(fts);
    Xs = normalize1(Xs);%Xs/max(max(abs(Xs)));
    ys = double(labels); %vector label
    clear fts; clear labels;
    load Clipart_vgg16_fc7.mat %target data
    Xt = double(fts);
    Xt = normalize1(Xt);%Xs/max(max(abs(Xs)));
    yt = double(labels); %vector label
    clear fts; clear labels;
elseif strcmp(db_name,'Art&Real')
    load Art_vgg16_fc7.mat %source data
    Xs = double(fts);
    Xs = normalize1(Xs);%Xs/max(max(abs(Xs)));
    ys = double(labels); %vector label
    clear fts; clear labels;
    load RealWorld_vgg16_fc7.mat %target data
    Xt = double(fts);
    Xt = normalize1(Xt);%Xs/max(max(abs(Xs)));
    yt = double(labels); %vector label
    clear fts; clear labels;
elseif strcmp(db_name,'Real&Art')
    load RealWorld_vgg16_fc7.mat %source data
    Xs = double(fts);
    Xs = normalize1(Xs);%Xs/max(max(abs(Xs)));
    ys = double(labels); %vector label
    clear fts; clear labels;
    load Art_vgg16_fc7.mat %target data
    Xt = double(fts);
    Xt = normalize1(Xt);%Xs/max(max(abs(Xs)));
    yt = double(labels); %vector label
    clear fts; clear labels;
elseif strcmp(db_name,'COIL1&COIL2')
    load COIL_1.mat %source data
    Xs = double(X_src)';
    Xs = normalize1(Xs);%Xs/max(max(abs(Xs)));
    ys = double(Y_src); %vector label
    Xt = double(X_tar)';
    Xt = normalize1(Xt);%Xs/max(max(abs(Xs)));
    yt = double(Y_tar); %vector label
    clear X_src;  clear Y_src;  clear X_tar;  clear Y_tar;
    
elseif strcmp(db_name,'COIL2&COIL1')
    load COIL_2.mat %source data
    Xs = double(X_src)';
    Xs = normalize1(Xs);%Xs/max(max(abs(Xs)));
    ys = double(Y_src); %vector label
    Xt = double(X_tar)';
    Xt = normalize1(Xt);%Xs/max(max(abs(Xs)));
    yt = double(Y_tar); %vector label
    clear X_src;  clear Y_src;  clear X_tar;  clear Y_tar;
    
end


[ndatat,~]      =     size(Xt);
R               =     randperm(ndatat);  %整数随机排列 行向量  row
exp_data.R      =   R;
% num_test        =  round(0.1*ndatat);
num_test        =  500;
test            =     Xt(R(1:num_test),:);
ytest           =     yt(R(1:num_test));
R(1:num_test)   =     [];
train           =     Xt(R,:);
train_ID = R;
Yt  =yt;
yt              =     yt(R);

Y=sparse(1:length(ys), double(ys), 1);
Y=full(Y);


% ytnew = knnclassify(train,Xs,ys); PWCF
mdl=fitcknn(Xs,ys);
ytnew=predict(mdl,train);
acc = length(find(ytnew==yt))/length(yt)
num_train       =     size(train,1);
if topNum == 0
    topNum      =     round(0.02*num_train);%set top Num as Two percent of  the number of train
    
end
DtrueTestTrain  =    distMat(test,train);
[~,idx]         =    sort(DtrueTestTrain,2);
idx             =    idx(:,1:topNum);
WtrueTestTrain  =    zeros(num_test,num_train);
for i=1:num_test
    WtrueTestTrain(i,idx(i,:)) =1;
end

if strcmp(param.retrieval, 'cross-domain')
    YS            =  repmat(ys,1,length(ytest));
    YT            =  repmat(ytest,1,length(ys));
elseif strcmp(param.retrieval, 'single-domain')
    YS            =  repmat(yt,1,length(ytest));
    YT            =  repmat(ytest,1,length(yt));
end
WTT           =  (YT==YS');

% size(Xs)
% size(Xt)
X=[Xs;Xt];
samplemean              = mean(X,1);
X = (double(X)-repmat(samplemean,size(X,1),1));
Xs                      = Xs-repmat(samplemean,size(Xs,1),1);
train                   = train-repmat(samplemean,size(train,1),1);
test                    = test-repmat(samplemean,size(test,1),1);

exp_data.db_data=X;
exp_data.num_test   = num_test;
exp_data.Xs         =   Xs ;
exp_data.train_ID   = train_ID;
exp_data.test       =   test;
exp_data.train      =   train;
exp_data.ys         =   ys ;
exp_data.yt         =   yt ;
exp_data.Yt         =   Yt ;
exp_data.train_all  =   [Xs;train];
exp_data.WTT           =WTT ;
exp_data.Y=Y;
exp_data.ytnew      =   ytnew ;
exp_data.WtrueTestTraining = WtrueTestTrain;
% exp_data.WtrueTestTrainingall=WtrueTestTrainingall;

end
