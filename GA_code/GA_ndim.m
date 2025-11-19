% 高维、动态变异率GA
% 长时间稳定结束算法，含复杂约束
%% 初始化
clear
close all
clc

%% 参数赋值
NP=100; % 种群数量
G=100;  % 最大遗传次数
L=20;   % 单维二进制数段长度
Dim=2;  % 维度
Max=[5,5];  % 定义域上限
Min=[-5,-5];    % 定义域下限
max_sta=inf;     % 最大稳定代数，稳定超过此代数将结束算法
max_err=30;     % 最大不符合约束条件次数
% 交叉率与变异率前往函数get_pc与get_pm修改

%% 参数错误检查
if length(Max)~=length(Min) || any(Max<Min)
    fprintf("[ERROR]定义域错误，请检查①上下限维度是否匹配②上限是否大于下限\n程序终止\n")
    return
end

%% 遗传算法
f=randi([0,1],NP,L*Dim);    % 随机获得初始种群
max_fit_rec=zeros(1,G);
%% 判断初始种群是否符合约束条件
for idx=1:NP
    while ~judge(b2d(f(idx,:),Min,Max,L))
        f(idx,:)=randi([0,1],1,L*Dim);
    end
    fprintf("初始种群约束条件纠错中，完成种群%d/%d\n",idx,NP)
end
for k=1:G
    tic
    %% 开始迭代
    for i=1:NP
        U=f(i,:);   % 获取第i条染色体
        x(i,:)=b2d(U,Min,Max,L);
        Fit(i)=aim(x(i,:));          % 计算每个样本适应度
    end
    [max_fit,max_fit_idx]=max(Fit);
    f_best=f(max_fit_idx,:);    % 此代最优个体的基因
    Fit1=(Fit-min(Fit))./(max_fit-min(Fit));    % 适应度归一化

    %% 计算稳定代数
    for idx=k-1:-1:1
        if max_fit_rec(idx)~=max_fit
            break
        end
    end
    sta_num=k-idx;  % 稳定代数
    
    %% 判断是否稳定超max_sta代
    if sta_num>=max_sta
        break
    end

    %% 基于轮盘赌选择法的复制操作
    sum_Fit=sum(Fit1);   % 计算种群中所有个体适应值的和
    fitvalue=Fit1/sum_Fit;   % 计算每个种群的选择概率
    fitvalue=(cumsum(fitvalue));    % 计算每个种群的累计概率
    ms=sort(rand(NP,1));    % 随机生成NP个(0,1)的值，并排序
    fitidx=1;     % 旧种群当前指针
    newidx=1;     % 新种群当前指针
    while newidx<=NP-1      % 随机复制，并使适应度大的遗传下去
        if ms(newidx) < fitvalue(fitidx)
            nf(newidx,:)=f(fitidx,:);   % 复制
            newidx=newidx+1;
        else
            fitidx=fitidx+1;
            if fitidx>NP
                break;
            end
        end
    end
    
    %% 基于概率的交叉操作
    for i=1:NP-2
        p=rand;     % 随机生成一个处于[0,1]的概率p
        if p<get_pc(sta_num)     % 满足交叉条件
            t=1;
            for trytime=1:max_err
                % 开始尝试交叉
                tnf=zeros(1,L*Dim);
                q=randi([0,1],1,L*Dim);     % 随机生成要交叉的基因位置
                for j=1:L*Dim
                    tnf(j)=nf(i+q(j),j);
                end
                if judge(b2d(tnf,Min,Max,L))  % 判断当前交叉结果是否符合约束条件
                    nf(i,:)=tnf(:);
                    break
                end
            end
        end
    end
    
    %% 基于概率的变异操作
    for i=1:round((NP-1)*get_pm(sta_num))   % 控制变异染色体总数
        h=randi([1,NP-1],1,1);    % 随机选择一个需要变异的染色体索引
        t=1;
        for trytime=1:max_err
            tnf=nf(h,:);
            for j=1:round(L*Dim*get_pm(sta_num)) % 控制变异基因数
                g=randi([1,L*Dim],1,1); % 随机需要变异的基因索引
                tnf(g)=~tnf(g);   % 取反
            end
            if judge(b2d(tnf,Min,Max,L))  % 判断当前变异结果是否符合约束条件
                nf(i,:)=tnf(:);
                break
            end
        end
    end

    %% 下一代预备
    f=nf;
    f(NP,:)=f_best; % 保留精英
    max_fit_rec(k)=max_fit;
    
    usetime=toc;
    fprintf("完成计算第%d代，进度%.6f，用时%.2fs，预计还需%.3fmin\n",k,k/G,usetime,usetime*(G-k)/60)
end

%% 收尾
for i=1:NP
    Fit(i)=aim(b2d(f(i,:),Min,Max,L));  % 计算每个样本适应度
end
[max_fit,max_fit_idx]=max(Fit);

%% 显示结果
fprintf("最大值:")
disp(max_fit)
fprintf("最大值点:")
disp(x(max_fit_idx,:))
fprintf('精度：x±')
disp((Max-Min)/(2^L-1));


%% 用户自定义函数
function fit=aim(x)
    % 目标函数
    fit=-(5*((x(2)-(x(1).^2)*5.1/(4*pi^2)+5/pi*x(1)-6).^2+10*(1-1/(8*pi))*cos(x(1)))+20);
end

function flag=judge(x)
    % 约束条件，满足条件返回1，否则返回0
    flag=1;

end


function pc=get_pc(sta_num)
    % 交叉率，sta_num-稳定代数
    pc=min(0.2+sta_num*0.05,0.9);   % 防止pc超过0.9
end

function pm=get_pm(sta_num)
    % 变异率，sta_num-稳定代数
    pm=min(0.2+sta_num*0.05,0.9);   % 防止pm超过0.9
end

function d=b2d(str,Min,Max,L)
    %格雷解码并将二进制转为定义域内的十进制
    %b2d(str,Xx,Xs,L),str-二进制行向量，Min-下限，Max-上限，L单维度二进制串长度
    Dim=length(Min);
    n=zeros(Dim,L);
    for k=1:Dim
        n(k,1)=str((k-1)*L+1);  % 格雷编码解码
        for j=2:L
            n(k,j)=xor(n(k,j-1),str((k-1)*L+j));    % xor—异或运算
        end
    end
    for di=1:Dim
        m(di)=0;
        for j=1:L   % 二进制转十进制
            m(di)=m(di)+n(di,L-j+1)*(2^(j-1));
        end
    end
    d=Min+m.*(Max-Min)./(2^L-1);    % 十进制解码至定义域
end

function str = d2b (arr,Min,Max,L)
    %格雷编码
    %d2b(arr,Xx,Xs,L),arr-十进制数据数组（行向量），Min-下限，Max-上限，L单维度二进制串长度；输出str为二进制行向量
    Dim=length(Min);
    str=zeros(1,Dim*L);
    m=round((arr-Min).*(2^L-1)./(Max-Min)); % 分散至二进制定义域
    n=double(boolean(dec2bin(m')-'0'));   % 十进制转二进制
    [~,col]=size(n);
    while col<L     % 首位补0至长为L
        n=[zeros(length(arr),1),n];
        [~,col]=size(n);
    end   
    for k=1:Dim % 格雷编码
        str((k-1)*L+1)=n(k,1);  % 首位不变
        str((k-1)*L+2:k*L)=xor(n(k,1:end-1),n(k,2:end));    % 异或运算
    end
end