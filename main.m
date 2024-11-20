% parameters
clear;
% time slot = 10s; frame=[25,60]frame/s; 
% size=[0.1e-3, 4]MB; z_k=[0.025, 2400]MB;
K=5;% GBD迭代次数
H=50;% 响应算法迭代次数
T=50;% 决策时隙的数量
N=12;% 用户集合 (1,2,..., N)
M=3;% 雾节点集合 (1,2,..., M)
c=M+1;% 云 指的是决策矩阵的第M+1个是云决策
Rc=(2*rand(1,N)+4.85)*(2^20); % 用户在云无线传输速率(bps) [4.85, 6.86] Mbps
Rm=(2*rand(1,N)+2.01)*(2^20);% 视频本地到雾实际传输速率(bps) [2.01, 4.01] Mbps
tn=30;% 云计算的往返时间(s)
p_cloud=2.605; % 云传输功率(Watt) 2605mJ/s
p_fog=1.22478;% 雾传输` 功率(Watt) 1224.78mJ/s
fc=20*10^9;% 云节点计算能力(Hz) 20GHz
fm=10*10^9;% 雾节点的计算能力(Hz) 10GHz
fn=(2*rand(1,N)+2)*10^9;
% fn=datasample([0.8,0.9,1,1.1,1.2], N)*10^9;% 本地计算能力(Hz) [0.8,0.9,1,1.1,1.2]GHz随机选择
k=10^(-27);% 本地计算消耗常数
c_cloud=9*10^(-10);% 云的付费模型(?/s)
c_fog=10*10^(-10);% 雾节点的付费模型(?/s)
alpha=0.04;% 雾节点折扣
B_max=800; % 预算(?)
LamdaT=0.99*rand(1,N)+0.01;% 时延权重
LamdaE=0.99*rand(1,N)+0.01;% 能耗权重
LamdaA=0.99*rand(1,N)+0.01;% 精度权重
%z_k=[50 100 800 1e3 2e3 4e3 10e3 20e3 30e3 40e3 50e3 100e3]*(2^13);
%z_k=[25e3 40e3 80e3 100e3 120e3 150e3 180e3 200e3 220e3 250e3]*(2^13);
z_k = randi([1e3,50e3],1,N)*(2^13);% 任务大小(bit) [1MB, 50MB]
Gamma=1300*rand(1,N)+1000;% 计算密度(cycle/bit) [10000, 23000]cycle/bit
Qe = 0;% 虚拟队列
V = 100;% 能耗和队列的平衡权重

beta_all = []; %压缩率
b_range = [0.7,0.9]; % 压缩率范围
b0 = randi([min(b_range*10),max(b_range*10)],1,N)/10;% 初始化压缩率
b_tra = min(b_range*100):5:max(b_range*100);% MP遍历压缩率
isFmin = 1; % MP方法选择，1为fmin，其他为遍历
isGBD = 0;% 1 for GBD, other for cast
if isGBD == 1
    beta_all = [beta_all;b0];% 压缩率
end

phi1=[]; phi2=[]; phi3=[]; % 精度
vl=normrnd(0.006,0.001,[1,N]); vs=normrnd(0.012,0.005,[1,N]);
vc=normrnd(0.013,0.005,[1,N]); v_all=[vl;vs;vc];
theta1=normrnd(50,10,[1,N]); theta2=normrnd(120,10,[1,N]);
theta3=normrnd(150,10,[1,N]); theta_all=[theta1;theta2;theta3];
phi1=[phi1;v_all(1,:).*log2(theta_all(1,:).*b0)]; % 本地精度
phi2=[phi2;v_all(2,:).*log2(theta_all(2,:).*b0)]; % 雾计算精度
phi3=[phi3;v_all(3,:).*log2(theta_all(3,:).*b0)]; % 云计算精度

a_x = [];% 决策空间
utility = []; %效益
budget = []; %预算
acc = []; %精度
energy = []; %能耗
time = []; %时延
target = []; %目标
UB=[]; LB=[];

for t = 1:T
    if (isGBD == 1)
        % GBD 求解
        [sol, ob, L, a, beta_k] = GBD_solver(V, Qe(end), beta_all(end,:),...
            N, M, c, Rc, Rm, tn, p_cloud, p_fog, fn, fc, fm, k, c_cloud, c_fog, ...
            alpha, LamdaT, LamdaE, LamdaA, z_k, Gamma, K, H, v_all, theta_all, ...
            b_range, b_tra, isFmin);   
        UB=[UB, ob];
        LB=[LB, L];
    else
        % cast algorithm
        [ob, a, beta_k] = primalP(V, Qe(end), ...
            N, M, c, Rc, Rm, tn, p_cloud, p_fog, fn, fc, fm, k, c_cloud, c_fog, ...
            alpha, LamdaT, LamdaE, LamdaA, z_k, Gamma, H, v_all, theta_all);  
    end
    % 数据处理
    phi1=[phi1;v_all(1,:).*log2(theta_all(1,:).*beta_k)];
    phi2=[phi2;v_all(2,:).*log2(theta_all(2,:).*beta_k)];
    phi3=[phi3;v_all(3,:).*log2(theta_all(3,:).*beta_k)];    
    a_x = [a_x;a];
    beta_all = [beta_all; beta_k];
    target = [target, ob];
    %penalty = [penalty, pen];
    z=z_k.*beta_k;
    Oc_a=[];Oc_u=[];Oc_t=[];
    Oc_e=[];Oc_q=[];n_m=[];
    for n=1:N
        if a(n)==0
            Oc_a(n)=phi1(end,n);
            Oc_e(n)=k*z(n)*Gamma(n)*(fn(n)^2);
            Oc_t(n)=z(n)*Gamma(n)/fn(n);
            Oc_q(n)=0;
            Oc_u(n)=LamdaT(n)*z(n)*Gamma(n)/fn(n)+LamdaE(n)*k*z(n)*Gamma(n)*fn(n)^2-LamdaA(n)*phi1(end,n);
        elseif a(n)==c
            Oc_a(n)=phi3(end,n);
            Oc_e(n)=p_cloud*z(n)/Rc(n);
            Oc_t(n)=z(n)/Rc(n)+tn+z(n)*Gamma(n)/fc;
            Oc_q(n)=c_cloud*z(n)*Gamma(n);
            Oc_u(n)=LamdaT(n)*(z(n)/Rc(n)+tn+z(n)*Gamma(n)/fc)+LamdaE(n)*p_cloud*z(n)/Rc(n)-LamdaA(n)*phi3(end,n);
        else
            for m = 1:M
                n_m(m) = sum(a == m);
            end
            v=n_m(a(n));
            Oc_a(n)=phi2(end,n);
            Oc_e(n)=p_fog*z(n)*v/Rm(n);
            Oc_t(n)=(1/Rm(n)+Gamma(n)/fm)*z(n)*v;
            Oc_q(n)=c_fog*(1-alpha*(v-1))*z(n)*Gamma(n);
            Oc_u(n)=LamdaT(n)*(1/Rm(n)+Gamma(n)/fm)*z(n)*v+LamdaE(n)*p_fog*z(n)*v/Rm(n)-LamdaA(n)*phi2(end,n);
        end
    end
    time = [time, sum(Oc_t)];
    energy = [energy, sum(Oc_e)];
    utility = [utility, -sum(Oc_u)];
    budget = [budget, sum(Oc_q)];
    acc = [acc, sum(Oc_a)];
    Qe = [Qe, max(Qe(end)+budget(end)-B_max,0)];
    disp(['ROUND: ', num2str(t)]);
    % 异常情况
    if budget(end) < 0 || budget(end) > 1.0e8
        disp("warning");
    end
end
  
%save('data12-50-12.mat');