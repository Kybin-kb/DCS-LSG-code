function [recall, precision, mAP, rec, pre, time, retrieved_list] = demo(exp_data, param,options, method)
ID.train = exp_data.train_ID;
WtrueTestTraining = exp_data.WTT ;
pos               = param.pos;
r                 = param.r;
train_data = exp_data.train_all;
test_data = exp_data.test;
db_data = exp_data.db_data;
NXs=size(exp_data.Xs,1);
all=size(exp_data.train_all,1);
% trueRank = exp_data.knn_p2;

[ntrain, D] = size(train_data);
%several state of art methods


% if strcmp(param.retrieval, 'cross-domain')  
%   rng(10495,'twister');  %ÖÖ×Ó
  switch(method)
    %% method proposed
          case 'SDH'
              addpath('./Method-SDH/');
              fprintf('......%s start ......\n\n', 'SDH');
              tic
              [B_trn,B_tst]=trainSDH(exp_data,param);
              toc
              fprintf('Train in %.3fs\n', toc);
              time=toc;
              if strcmp(param.retrieval, 'cross-domain')
                  B_trn           =     B_trn(1:NXs,:);
              elseif strcmp(param.retrieval, 'single-domain')
                  B_trn           =    B_trn(NXs+1:all,:);
              end
              
        case 'DAPH' 
            addpath('./Method-DAPH/');
            fprintf('......%s start ......\n\n', 'DAPH');
          t1 = clock;
            [B_trn,B_tst]=trainDAPH(exp_data,param);
          t2 = clock;
          t = etime(t2,t1);
          fprintf('Train in %.3fs\n', t);
          time=t;

        case 'DAPH*' 
            addpath('./Method-DAPH1/');
            fprintf('......%s start ......\n\n', 'DAPH*');
            t1 = clock;
            [B_trn,B_tst]=trainDAPH1(exp_data,param);
          t2 = clock;
          t = etime(t2,t1);
          fprintf('Train in %.3fs\n', t);
          time=t;

      case 'Ours'
          addpath('./myMethod/');
          fprintf('\n......%s start ......\n\n', 'DCS-LSG');
          t1 = clock;
          [B_trn,B_tst]=DCSLSG(exp_data,param);
          t2 = clock;
          t = etime(t2,t1);
          fprintf('Train in %.3fs\n', t);
          time=t;
          
       case 'SGHL'
          addpath('./Method-LGSCH/');
          fprintf('\n......%s start ......\n\n', 'SGHL');
          t1 = clock;
          [B_trn,B_tst]=trainLGSCH(exp_data,param);
          t2 = clock;
          t = etime(t2,t1);
          fprintf('Train in %.3fs\n', t);
          time=t;
          
      case 'Ours1'
          addpath('./myMethod1/');
          fprintf('\n......%s start ......\n\n', 'LGSCH1');
          tic
          [B_trn,B_tst]=LGSCH1(exp_data,param);
          toc
          fprintf('Train in %.3fs\n', toc);
          time=toc;
          
      case 'Ours2'
          addpath('./myMethod2/');
          fprintf('\n......%s start ......\n\n', 'LGSCH2');
          tic
          [B_trn,B_tst]=LGSCH2(exp_data,param);
          toc
          fprintf('Train in %.3fs\n', toc);
          time=toc;
          
      case 'Ours3'
          addpath('./myMethod3/');
          fprintf('\n......%s start ......\n\n', 'LGSCH3');
          tic
          [B_trn,B_tst]=LGSCH3(exp_data,param);
          toc
          fprintf('Train in %.3fs\n', toc);
          time=toc;
          
      case 'Ours4'
          addpath('./myMethod4/');
          fprintf('\n......%s start ......\n\n', 'LGSCH4');
          tic
          [B_trn,B_tst]=LGSCH4(exp_data,param);
          toc
          fprintf('Train in %.3fs\n', toc);
          time=toc;
          
          
      case 'DA'
          addpath('./Method-DA/');
          fprintf('\n......%s start ......\n\n', 'DA-');
          tic
          [B_trn,B_tst]=trainDA(exp_data,param,options);
          toc      
          fprintf('Train in %.3fs\n', toc);
          time=toc;
      case 'DAY'
          addpath('./Method-DA/');
          fprintf('\n......%s start ......\n\n', 'DAY-');
          [options,list]=getGobalOptions('lambda',1e-2,'alpha',0,'gamma',1e-2,'beta',1e-4,'theta',1e4);
          tic
          [B_trn,B_tst]=trainDA(exp_data,param,options);
          toc      
          fprintf('Train in %.3fs\n', toc);
          time=toc;          
      case 'ORTH'
          addpath('./Method-ORTH/');
          fprintf('\n......%s start ......\n\n', 'ORTH-');
          tic
          [B_trn,B_tst]=trainORTH(exp_data,param,options);
          toc      
          fprintf('Train in %.3fs\n', toc);
          time=toc;
      case 'TWOSTEP'
          addpath('./Method-TWOSTEP/');
          fprintf('\n......%s start ......\n\n', 'TWOSTEP-');
          tic
          [B_trn,B_tst]=trainTWOSTEP(exp_data,param,options);
          toc      
          fprintf('Train in %.3fs\n', toc);
          time=toc;      
      case 'Y'
          addpath('./Method-Y/');
          fprintf('\n......%s start ......\n\n', 'Y-');
          tic
          [B_trn,B_tst]=trainY(exp_data,param,options);
          toc      
          fprintf('Train in %.3fs\n', toc);
          time=toc;   
          
      case 'PWCF'
          addpath('./Method-PWCF/');
          fprintf('......%s start ......\n\n', 'PWCF');
          t1 = clock;
          [B_trn,B_tst]=trainPWCF(exp_data,param);
          t2 = clock;
          t = etime(t2,t1);
          fprintf('Train in %.3fs\n', t);
          time=t;

      case 'ITQ'
          addpath('./Method-ITQ/');
          addpath('./Method-PCAH/');
          fprintf('......%s start...... \n\n', 'PCA-ITQ');
          tic;
          ITQparam.nbits = param.nbits;
          %ITQparam =  trainPCAH(db_data, ITQparam);
          ITQparam =  trainPCAH(train_data, ITQparam);
          ITQparam = trainITQ(train_data, ITQparam);
          [B_trn, ~] = compressITQ(train_data, ITQparam);
          [B_tst, ~] = compressITQ(test_data, ITQparam);
          toc;
          time=toc;

          if strcmp(param.retrieval, 'cross-domain')
              B_trn           =     B_trn(1:NXs,:);
          elseif strcmp(param.retrieval, 'single-domain')
              B_trn           =    B_trn(NXs+1:all,:);
          end

          %[B_db, ~] = compressITQ(db_data, ITQparam);
          clear db_data ITQparam;
      % SGH hashing
      case 'SGH'
          addpath('./Method-SGH/');
          fprintf('......%s start...... \n\n', 'SGH');
          tic;
          %sample = randperm(ndata);
          % Kernel parameter
          s = RandStream('mt19937ar','Seed',0);
          sample = randperm(s, ntrain);
          m = 300;
          bases = train_data(sample(1:m),:);
          SGHparam.r = param.nbits;
          [Wx, KXTrain, para] = trainSGH(train_data, bases,SGHparam.r);
          B_trn = (KXTrain*Wx > 0);
          % construct KXTest
          KTest = distMat(test_data,bases);
          KTest = KTest.*KTest;
          KTest = exp(-KTest/(2*para.delta));
          [num_testing, D] = size(test_data);
          KXTest = KTest-repmat(para.bias,num_testing,1);
          B_tst = (KXTest*Wx > 0);    
          toc;
          time=toc;

          if strcmp(param.retrieval, 'cross-domain')
              B_trn           =     B_trn(1:NXs,:);
          elseif strcmp(param.retrieval, 'single-domain')
              B_trn           =    B_trn(NXs+1:all,:);
          end
           clear db_data SGHparam;
          % Locality sensitive hashing (LSH)
      case 'LSH'
          addpath('./Method-LSH/');
          fprintf('......%s start ......\n\n', 'LSH');
          tic;
          LSHparam.nbits = param.nbits;
          LSHparam.dim = D;
          LSHparam = trainLSH(LSHparam);
          [B_trn, ~] = compressLSH(train_data, LSHparam);
          [B_tst, ~] = compressLSH(test_data, LSHparam);
          toc;
          time=toc;
          if strcmp(param.retrieval, 'cross-domain')
              B_trn           =     B_trn(1:NXs,:);
          elseif strcmp(param.retrieval, 'single-domain')
              B_trn           =    B_trn(NXs+1:all,:);
          end
          %[B_db, ~] = compressLSH(db_data, LSHparam);
          clear db_data LSHparam;
          
          % Spetral hashing
      case 'SH'
          addpath('./Method-SH/');
          addpath('./Method-PCAH/');
          fprintf('......%s start...... \n\n', 'SH');   
          tic;
          SHparam.nbits = param.nbits;
          SHparam =  trainPCAH(db_data, SHparam);
          SHparam = trainSH(train_data, SHparam);
          [B_trn, ~] = compressSH(train_data, SHparam);
          [B_tst, ~] = compressSH(test_data, SHparam);
          toc;
          time=toc;
          if strcmp(param.retrieval, 'cross-domain')
              B_trn           =     B_trn(1:NXs,:);
          elseif strcmp(param.retrieval, 'single-domain')
              B_trn           =    B_trn(NXs+1:all,:);
          end
          %[B_db, ~] = compressITQ(db_data, ITQparam);
                
      % Density sensitive hashing
      case 'DSH'
          addpath('./Method-DSH/');
          fprintf('......%s start ......\n\n', 'DSH');
          tic;
          DSHparam.nbits = param.nbits;
          DSHparam = trainDSH(train_data, DSHparam);
          [B_trn, ~] = compressDSH(train_data, DSHparam);
          [B_tst, ~] = compressDSH(test_data, DSHparam);
          toc;
          time=toc;
          if strcmp(param.retrieval, 'cross-domain')
              B_trn           =     B_trn(1:NXs,:);
          elseif strcmp(param.retrieval, 'single-domain')
              B_trn           =    B_trn(NXs+1:all,:);
          end
          clear db_data DSHparam;
          % unsupervised sequential projection learning based hashing
      case 'GTH'
          addpath('./Method-GTH-g/');
          fprintf('......%s start ......\n\n', 'GTH-g');     
          pos               = param.pos;
          t1 = clock;
          [B_trn,B_tst]=trainGTHg(exp_data,param);
          t2 = clock;
          t = etime(t2,t1);
          fprintf('Train in %.3fs\n', t);
          time=t;
          
      case 'GTH-h'
          addpath('./Method-GTH-h/');
          fprintf('......%s start ......\n\n', 'GTH-h');
          pos               = param.pos;
          tic
          [B_trn,B_tst]=trainGTHh(exp_data,param);
          toc
          fprintf('Train in %.3fs\n', toc);
          time=toc;
  end

% elseif strcmp(param.retrieval, 'single-domain')
%   switch(method)
%     %% method proposed
% 
%         case 'DAPH'  
%         addpath('./Method-DAPH/');
% 	    fprintf('......%s start ......\n\n', 'DAPH');
%           tic
%         [B_trn,B_tst]=trainDAPH(exp_data,param);
%         toc
%         fprintf('Train in %.3fs\n', toc);
%         
%         case 'DAPH*'  
%         addpath('./Method-DAPH1/');
% 	    fprintf('......%s start ......\n\n', 'DAPH*');
%           tic
%         [B_trn,B_tst]=trainDAPH1(exp_data,param);
%         toc
%         fprintf('Train in %.3fs\n', toc);
%   end
% end

% compute Hamming metric and compute recall precision
Dhamm = hammingDist(B_tst, B_trn);
[~, rank] = sort(Dhamm, 2, 'ascend');
clear B_tst B_trn;
choice = param.choice;
% Dhamm
switch(choice)
    case 'evaluation_PR_MAP'
        clear train_data test_data;
        [recall, precision, ~] = recall_precision(WtrueTestTraining, Dhamm);
        [rec, pre]= recall_precision5(WtrueTestTraining, Dhamm, pos); % recall VS. the number of retrieved sample
        [mAP] = area_RP(recall, precision);
        retrieved_list = [];
    case 'evaluation_PR'
        clear train_data test_data;
        eva_info = eva_ranking(rank, trueRank, pos);
        rec = eva_info.recall;
        pre = eva_info.precision;
        recall = [];
        precision = [];
        mAP = [];
        retrieved_list = [];
    case 'visualization'
        num = param.numRetrieval;
        retrieved_list =  visualization(Dhamm, ID, num, train_data, test_data); 
        recall = [];
        precision = [];
        rec = [];
        pre = [];
        mAP = [];
end

end
