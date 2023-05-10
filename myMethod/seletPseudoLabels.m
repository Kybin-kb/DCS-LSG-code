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
    
    %% ��ѡp1��p2Ԥ��classһ������
    [~,I1] = max(probMatrix1');
    [~,I2] = max(probMatrix2');
    samePredict = find(I1 == I2); % P1 P2Ԥ����ȵ��±꼯��
    prob1 = prob(samePredict);  % ȡ����ЩԤ��һ�������ĸ���
    predLabels1 = predLabels(samePredict);  % ȡ����ЩԤ��һ��������Ԥ���ǩ
    
    p=iter/num_iter;
    p = max(p,0);
    [sortedProb,index] = sort(prob1);  % ��Ԥ��һ��������Ԥ��������򣬵õ���index��ӦsamePredict���±�
    sortedPredLabels = predLabels1(index);
    trustable = zeros(1,length(prob1));
    %% ��ÿ�����а���Ԥ����������ƽ��˼����ѡ����
    for i = 1:num_class
        ntc = length(find(predLabels==i));
        ntc_same = length(find(predLabels1 == i));
        % Ҫ��Ԥ��һ���������ҵ�ǰclass��ע�����indexҪһ�£����ڶ���samePredict�е��±�
        thisClassProb = sortedProb(sortedPredLabels==i);
        if length(thisClassProb)>0
            %��ÿ�����а���Ԥ����������ƽ��˼����ѡ��min(iter/num_iter * nc, sameDc)������
            minProb = thisClassProb(max(ntc_same-(floor(p*ntc)+1) , 1));
            % �ҳ�Ԥ��һ��������Ԥ��ֵ������СԤ����ֵ��������ע�⣬�õ�����samePredict�е��±�
            trustable = trustable+ (prob1>minProb).*(predLabels1==i);
        end
    end
    % �ҵ�������ӦĿ����������index
    true_index = samePredict(trustable==1);
    pseudoLabels = predLabels;
    trustable = zeros(1, length(prob));
    trustable(true_index) = 1;
    Xt0 = domainT_features(true_index,:);
    pseudoLabels = pseudoLabels(:,true_index);
%     pseudoLabels(~trustable) = -1;
  
end