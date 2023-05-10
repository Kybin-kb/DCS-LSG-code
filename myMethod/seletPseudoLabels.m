function [pseudoLabels,predLabels,Xt0] = seletPseudoLabels(domainS_features,domainT_features,domainS_labels,num_class,iter,num_iter,P1,P2)
    domainS_features = domainS_features';
    domainT_features = domainT_features';
    domainS_proj = domainS_features*P1;%n*c
    domainT_proj = domainT_features*P2;
    proj_mean = mean([domainS_proj;domainT_proj]);
    domainS_proj = domainS_proj - repmat(proj_mean,[size(domainS_proj,1) 1 ]);
    domainT_proj = domainT_proj - repmat(proj_mean,[size(domainT_proj,1) 1 ]);
    domainS_proj = L2Norm(domainS_proj);
    domainT_proj = L2Norm(domainT_proj);
    %% distance to class means
    classMeans = zeros(num_class,num_class);
    for i = 1:num_class
        classMeans(i,:) = mean(domainS_proj(domainS_labels==i,:));
    end
    classMeans = L2Norm(classMeans);
    distClassMeans = EuDist2(domainT_proj,classMeans);
    targetClusterMeans = vgg_kmeans(double(domainT_proj'), num_class, classMeans')';
    targetClusterMeans = L2Norm(targetClusterMeans);
    distClusterMeans = EuDist2(domainT_proj,targetClusterMeans);
    expMatrix = exp(-distClassMeans);
    expMatrix2 = exp(-distClusterMeans);
    probMatrix1 = expMatrix./repmat(sum(expMatrix,2),[1 num_class]);
    probMatrix2 = expMatrix2./repmat(sum(expMatrix2,2),[1 num_class]);
    
    probMatrix = probMatrix1 * (1-iter./num_iter) + probMatrix2 * iter./num_iter;
    [prob,predLabels] = max(probMatrix');
    
    %% 挑选p1和p2预测class一样的类
    [~,I1] = max(probMatrix1');
    [~,I2] = max(probMatrix2');
    samePredict = find(I1 == I2); % P1 P2预测相等的下标集合
    prob1 = prob(samePredict);  % 取出这些预测一致样本的概率
    predLabels1 = predLabels(samePredict);  % 取出这些预测一致样本的预测标签
    
    p=iter/num_iter;
    p = max(p,0);
    [sortedProb,index] = sort(prob1);  % 对预测一致样本的预测概率排序，得到的index对应samePredict的下标
    sortedPredLabels = predLabels1(index);
    trustable = zeros(1,length(prob1));
    %% 从每个类中按照预设条件和类平衡思想挑选样本
    for i = 1:num_class
        ntc = length(find(predLabels==i));
        ntc_same = length(find(predLabels1 == i));
        % 要从预测一致样本中找当前class，注意二者index要一致，现在都是samePredict中的下标
        thisClassProb = sortedProb(sortedPredLabels==i);
        if length(thisClassProb)>0
            %从每个类中按照预设条件和类平衡思想挑选出min(iter/num_iter * nc, sameDc)个样本
            minProb = thisClassProb(max(ntc_same-(floor(p*ntc)+1) , 1));
            % 找出预测一致样本中预测值大于最小预测阈值的样本，注意，得到的是samePredict中的下标
            trustable = trustable+ (prob1>minProb).*(predLabels1==i);
        end
    end
    % 找到真正对应目标域样本的index
    true_index = samePredict(trustable==1);
    pseudoLabels = predLabels;
    trustable = zeros(1, length(prob));
    trustable(true_index) = 1;
    Xt0 = domainT_features(true_index,:);
    pseudoLabels = pseudoLabels(:,true_index);
%     pseudoLabels(~trustable) = -1;
  
end