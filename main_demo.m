close all; clear all; clc;
addpath('./utils/');
addpath(genpath('./liblinear-2.30'));
result_URL = './results/';
if ~isdir(result_URL)
    mkdir(result_URL);
end
options_URL='./options/';
if ~isdir(options_URL) %是否有文件夹  dir：列出当前文件夹中的文件和文件夹。
    mkdir(options_URL);%创建文件夹
end
save_URL='./save/';
if ~isdir(save_URL)
    mkdir(save_URL);
end

db_name   =   'MNIST&USPS';
% db_name   = 'VOC2007&Caltech101';%domain adaptaation datasets 
% db_name   = 'Caltech256&ImageNet';

%  db_name   = 'Product&Real';
%  db_name   = 'Real&Product';
%  db_name   = 'Clipart&Real';
%  db_name   = 'Real&Clipart';
%  db_name   = 'Art&Real';
%  db_name   = 'Real&Art';



choose_data = 'yes';% 'yes' or 'no'
param.choice= 'evaluation_PR_MAP';%  evaluation_PR
param.retrieval= 'cross-domain'; %'single-domain' or  'cross-domain'
% loopnbits = [16 32 48 64 96 128];
loopnbits = [64]; 
runtimes = 1; % change 10 times to make the rusult more smooth in the paper

% param.pos = [1:10:40 50:20:130];  % The number of retrieved samples: Recall-The number of retrieved samples curve
param.pos = [1:10:40 50:20:400];
% param.pos = [1:10:40 50:50:1000];

hashmethods = {'Ours'};
% hashmethods = {'GTH','GTH-h','DAPH','DAPH*','PWCF','SGHL','DCSS-LSG'};
% hashmethods = {'SH','SGH','LSH','ITQ','GTH','PWCF','Ours'};
% hashmethods = {'SH','SGH','LSH','ITQ','GTH','DAPH','DAPH*','PWCF','Ours'};
% hashmethods = {'SH','SGH','LSH','ITQ','Ours'};
% hashmethods = {'DCSS-LSG'};
% hashmethods = {'Ours','Ours1','Ours2','Ours3'};
% hashmethods = {'GTH','DCSS-LSG'};
% hashmethods = {'Ours','Ours1'};

nhmethods = length(hashmethods);
retrieval=param.retrieval;
query_ID = []; 
    diary on;

for k = 1:runtimes
    if strcmp(choose_data,'yes')
        fprintf('The %d run time, start constructing data\n\n', k); 
        exp_data = construct_dataset(db_name,param,runtimes); 
        fprintf('Constructing data finished\n\n');
    else
        load('exp_data.mat', 'exp_data')
    end
    result_name = [result_URL 'final_' db_name '_result' '.mat'];
    save_name = [save_URL db_name '_exp_data' '.mat'];
    options_name=[options_URL  db_name '_result'  '.mat'];
    
    %参数
     [options,list]=getGobalOptions('lambda',1e-2,'alpha',1e-2,'gamma',1e-2,'beta',1e-4,'theta',1e4);%[1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,5e-3,1e-2]
        
     for i =1:length(loopnbits)
            fprintf('\n======start %d bits encoding======\n\n', loopnbits(i));
            param.r = loopnbits(i);   
            param.nbits = loopnbits(i);
            param.query_ID = query_ID;
            for j = 1:nhmethods
                [recall{k}{i, j}, precision{k}{i, j}, mAP{k}{i,j}, rec{k}{i, j}, pre{k}{i, j},time{k}{i,j},~] = demo(exp_data, param, options,hashmethods{1, j});
%                 [recall{k}{i, o}, precision{k}{i, o}, mAP{k}{i,o}, rec{k}{i, o}, pre{k}{i, o}, time{k}{i,o},~] = demo(exp_data, param, options,hashmethods{1, j});
            end
        end  
    clear exp_data;  
end

% plot attribution
line_width = 1.5;
marker_size = 6;
xy_font_size = 14;
legend_font_size = 12;
linewidth = 1.6;
title_font_size = xy_font_size;
% average MAP
for j = 1:nhmethods
    for i =1: length(loopnbits)
        tmp = zeros(size(mAP{1, 1}{i,j}));
        for k = 1:runtimes
            tmp = tmp+mAP{1, k}{i, j};            
        end      
        MAP{i, j} = tmp/runtimes;      
    end
    clear tmp;
end
MAP

% % average MAP
% for o=1:size(list,1)
    % for i =1: length(loopnbits)
        % tmp = zeros(size(mAP{1, 1}{i,o}));
        % for k = 1:runtimes
            % tmp = tmp+mAP{1, k}{i, o};           
        % end       
        % MAP{i, o} = tmp/runtimes;      
        % options(o).map16=MAP{i,o};
    % end
    % clear tmp;
% end
% MAP
% for o=1:size(list,1)
    % for i =1: length(loopnbits)
        % if i==1
            % options(o).MAP16=MAP{i,o};
        % elseif  i==2
            % options(o).MAP32=MAP{i,o};
        % elseif i==3
             % options(o).MAP48=MAP{i,o};
        % elseif i==4
            % options(o).MAP64=MAP{i,o};        
        % elseif i==5
            % options(o).MAP96=MAP{i,o};        
        % else 
            % options(o).MAP128=MAP{i,o};
        % end
    % end   
% end

% save result

save(result_name, 'precision','pre', 'recall', 'rec', 'MAP', 'mAP', 'hashmethods', 'nhmethods', 'loopnbits','options','retrieval','list','db_name','time');
    diary off;
choose_bits  = 1; % i: choose the bits to show
choose_times = 1; % k is the times of run times
%% show recall vs. the number of retrieved sample.
figure('Color', [1 1 1]); hold on;

for j = 1: nhmethods
    pos = param.pos;
    recc = rec{choose_times}{choose_bits, j};
    %p = plot(pos(1,1:posEnd), recc(1,1:posEnd));
    p = plot(pos(1,1:end), recc(1,1:end));
    color = gen_color(j);
    marker = gen_marker(j);
    set(p,'Color', color)
    set(p,'Marker', marker);
    set(p,'LineWidth', line_width);
    set(p,'MarkerSize', marker_size);
end

str_nbits =  num2str(loopnbits(choose_bits));
set(gca, 'linewidth', linewidth);
h1 = xlabel('The number of retrieved samples');
h2 = ylabel(['Recall @ ', str_nbits, ' bits']);
title(db_name, 'FontSize', title_font_size);
set(h1, 'FontSize', xy_font_size);
set(h2, 'FontSize', xy_font_size);
axis square;
hleg = legend(hashmethods);
set(hleg, 'FontSize', legend_font_size);
set(hleg,'Location', 'best');
box on;
grid on;
hold off;

%% show precision vs. the number of retrieved sample.
figure('Color', [1 1 1]); hold on;

for j = 1: nhmethods
    pos = param.pos;
    prec = pre{choose_times}{choose_bits, j};
    %p = plot(pos(1,1:posEnd), recc(1,1:posEnd));
    p = plot(pos(1,1:end), prec(1,1:end));
    color = gen_color(j);
    marker = gen_marker(j);
    set(p,'Color', color)
    set(p,'Marker', marker);
    set(p,'LineWidth', line_width);
    set(p,'MarkerSize', marker_size);
end


str_nbits =  num2str(loopnbits(choose_bits));
set(gca, 'linewidth', linewidth);
h1 = xlabel('The number of retrieved samples');
h2 = ylabel(['Precision @ ', str_nbits, ' bits']);
title(db_name, 'FontSize', title_font_size);
set(h1, 'FontSize', xy_font_size);
set(h2, 'FontSize', xy_font_size);
axis square;
hleg = legend(hashmethods);
set(hleg, 'FontSize', legend_font_size);
set(hleg,'Location', 'best');
box on;
grid on;
hold off;

%% show precision vs. recall , i is the selection of which bits.
figure('Color', [1 1 1]); hold on;

for j = 1: nhmethods
    p = plot(recall{choose_times}{choose_bits, j}, precision{choose_times}{choose_bits, j});
    color=gen_color(j);
    marker=gen_marker(j);
    set(p,'Color', color)
    set(p,'Marker', marker);
    set(p,'LineWidth', line_width);
    set(p,'MarkerSize', marker_size);
end

str_nbits = num2str(loopnbits(choose_bits));
h1 = xlabel(['Recall @ ', str_nbits, ' bits']);
h2 = ylabel('Precision');
title(db_name, 'FontSize', title_font_size);
set(h1, 'FontSize', xy_font_size);
set(h2, 'FontSize', xy_font_size);
axis square;
hleg = legend(hashmethods);
set(hleg, 'FontSize', legend_font_size);
set(hleg,'Location', 'best');
set(gca, 'linewidth', linewidth);
box on;
grid on;
hold off;

%% show mAP. This mAP function is provided by Yunchao Gong
figure('Color', [1 1 1]); hold on;
for j = 1: nhmethods
    map = [];
    for i = 1: length(loopnbits)
        map = [map, MAP{i, j}];
    end
    p = plot(log2(loopnbits), map);
    color=gen_color(j);
    marker=gen_marker(j);
    set(p,'Color', color);
    set(p,'Marker', marker);
    set(p,'LineWidth', line_width);
    set(p,'MarkerSize', marker_size);
end

h1 = xlabel('Number of bits');
h2 = ylabel('mean Average Precision (mAP%)');
title(db_name, 'FontSize', title_font_size);
set(h1, 'FontSize', xy_font_size);
set(h2, 'FontSize', xy_font_size);
axis square;
set(gca, 'xtick', log2(loopnbits));
set(gca, 'XtickLabel', {'16','32','48','64','96','128'});
set(gca, 'linewidth', linewidth);
hleg = legend(hashmethods);
set(hleg, 'FontSize', legend_font_size);
set(hleg, 'Location', 'best');
box on;
grid on;
hold off;
