function [is_fs, mu, L_k, dL_k, UB, penalty, a] = primal_solver ( V, Q, gamma, ...
    N, M, c, Rc, Rm, tn, p_cloud, p_fog, fn, fc, fm, k, c_cloud, c_fog, alpha, ...
    LamdaT, LamdaE, LamdaA, z_k, Gamma, H, va, theta, phi1, phi2, phi3)
% init to avoid the complain of matlab ...
a_n = [];% 决策空间
T = H;
pen=zeros(1,N);
Oc_q=zeros(1,N);
Oc=zeros(1,N);
beta = gamma; % 压缩率
z=z_k.*beta;
a = zeros(T, N); % 每个用户选择本地计算
n_m = zeros(1, M);
allowed=zeros(1,M+1); % 判断矩阵
% solve primal problem with best response %
for t = 1:T
    % 雾节点向用户广播其负载
    for m = 1:M
        n_m(m) = sum(a(t, :) == m);
    end
    allowed=zeros(1,M+1);
    % 每个用户计算最佳响应函数并更新任务卸载策略
    for n = 1:N
        bn = best_response(Q, V, M, c, Rc, Rm, tn, p_cloud, p_fog, fn, fc, fm, k, c_cloud, c_fog, alpha, LamdaT, LamdaE, LamdaA, z, Gamma, n, t, a, n_m, phi1, phi2, phi3);
        if ismember(0, bn)
            a(t+1, n) = 0;
        elseif ismember(c, bn)
            a(t+1, n) = c;
        else
            % 发送请求更新的信息至雾节点以竞争更新任务卸载策略的机会
            compare=allowed;
            allowed=update_allowed(M,bn,allowed,a,t,n);
            if  isequal(compare,allowed)
                a(t+1, n) = a(t, n);
            else
                a(t+1, n) = allowed(M+1);
            end
        end
    end

end

a = a(T,:); % 决策
for n=1:N
    if a(n)==0
        pen(n)=LamdaT(n)*z(n)*Gamma(n)/fn(n)+LamdaE(n)*k*z(n)*Gamma(n)*(fn(n)^2)-LamdaA(n)*phi1(n);
        Oc(n)=V*pen(n);
    elseif a(n)==c
        [pen(n),Oc_q(n)]=Cloud_processing(LamdaT,p_cloud,z,Gamma,Rc,fc,LamdaE,c_cloud,LamdaA,tn,n,phi3);
        Oc(n)=Q*Oc_q(n) + V*pen(n);
    else
        for m = 1:M
            n_m(m) = sum(a == m);
        end
        v=n_m(a(n));
        pen(n)=LamdaT(n)*(1/Rm(n)+Gamma(n)/fm)*z(n)*v+LamdaE(n)*p_fog*z(n)*v/Rm(n)-LamdaA(n)*phi2(n);
        Oc_q(n)=c_fog*(1-alpha*(v-1))*z(n)*Gamma(n);
        Oc(n)=Q*Oc_q(n) + V*pen(n);
    end
end

UB=sum(Oc);
penalty=sum(pen);
status = "Solved"; % primalP 一定可解


x_n = zeros(M+2, N); % M+2xN的矩阵
for i = 1:length(a)
    x_n(a(i)+1, i) = 1;
end
disp(['a: ', num2str(a)]);

%%
syms xv [M+2,N];
syms bv [1 N];
temp=sum(xv(2:M+1,:),2);
O1_n=V*(LamdaT.*z_k.*bv.*Gamma./fn+LamdaE.*k.*z_k.*bv.*Gamma.*(fn.^2)-LamdaA.*(va(1,:).*log2(theta(1,:).*bv)));
O2_n=Q*c_fog*(1-alpha*(temp-1)).*z_k.*bv.*Gamma+V*(LamdaT.*(1./Rm+Gamma./fm).*z_k.*bv.*temp+LamdaE*p_fog.*z_k.*bv./Rm.*temp-LamdaA.*(va(2,:).*log2(theta(2,:).*bv)));
O3_n=Q*c_cloud.*z_k.*bv.*Gamma+V*(LamdaT.*(z_k.*bv./Rc+tn+z_k.*bv.*Gamma./fc)+LamdaE*p_cloud.*z_k.*bv./Rc-LamdaA.*va(3,:).*log2(theta(3,:).*bv));
opt_func=sum(xv(1,:).*O1_n)+sum(sum(sum(xv(2:M+1,:).*O2_n)))+sum(xv(M+2,:).*O3_n);
cst_func=sum(sum(xv)-1);

if (status ~= "Solved")
    is_fs = false; mu = []; cut = [];
else
    disp(['Current Optimal Value : ', num2str(UB)])
    is_fs = true;

    % Lagrange Mutliplier %
    % mu = find_mu_primal(N,M,x_n, beta, 0, opt_func, cst_func);
    mu = zeros(M+2,N);

    % Lagrange Function %
    g = subs(cst_func, xv, x_n);
    P = subs(opt_func, xv, x_n);
    if all(g.*mu==0)
        L = P;
    else
        L = P + g.*mu;
    end

    % Lagrange Function %
    g_k = subs(subs(cst_func, xv, x_n), bv, beta);
    P_k = double(subs(subs(opt_func, xv, x_n), bv, beta));
    if all(g_k.*mu==0)
        L_k = P_k;
    else
        L_k = P_k + g_k.*mu;
    end
    % Optimality cut %
    syms eta
    for n=1:N
        dL_k(n) = double(subs(diff(L,bv(n)), bv, beta)); %1xN
    end
    Opt  = L_k + dL_k.*(bv-beta);
    cut  = Opt - eta; % <= 0
    %disp(cut);

end
end

function bn = best_response(Q, V, M, c, Rc, Rm, tn, p_cloud, p_fog, fn, fc, fm, ...
    k, c_cloud, c_fog, alpha, LamdaT, LamdaE, LamdaA, z, Gamma, n, t, a, ...
    n_m, phi1, phi2, phi3)
% 计算用户n的最佳响应函数
bn=[];
O = zeros(1,M+2);
O1_u=Local_processing(LamdaT,LamdaE,LamdaA, k,z,Gamma,fn,n,phi1);
[O3_u,O3_q]=Cloud_processing(LamdaT,p_cloud,z,Gamma,Rc,fc,LamdaE,c_cloud,LamdaA,tn,n,phi3);
O1=V*O1_u;
O3=Q*O3_q+V*O3_u;
for p = 1:(M+2)
    if (p-1)==0
        O(p)=O1;
    elseif (p-1)==c
        O(p)=O3;
    else
        [O2_u,O2_q]=Fog_proccessing(LamdaT,LamdaE,LamdaA,z,Gamma,Rm,c_fog,fm,p_fog,alpha,a,n,t,p,n_m,M,phi2);
        O2=Q*O2_q+V*O2_u;
        O(p)=O2;
    end
end
for p = 1:(M+2)
    if O(p) == min(O)
        bn = [bn p-1];
    end
end
end

function [O2_u,O2_q]=Fog_proccessing(LamdaT,LamdaE,LamdaA,z,Gamma,Rm,c_fog,fm,p_fog,alpha,a,n,t,p,n_m,M,phi2)
d=n_m;
q=a(t,n);%第n个人在第t轮选择的方案
if q>=1&&q<=M
    d(q)=d(q)-1;
end
v=d(p-1)+1;
O2_u=LamdaT(n)*(1/Rm(n)+Gamma(n)/fm)*z(n)*v+LamdaE(n)*p_fog*z(n)*v/Rm(n)-LamdaA(n)*phi2(n);
O2_q=c_fog*(1-alpha*(v-1))*z(n)*Gamma(n);
end

function O1_u=Local_processing(LamdaT,LamdaE,LamdaA, k,z,Gamma,fn,n,phi1)
O1_u=LamdaT(n)*z(n)*Gamma(n)/fn(n)+LamdaE(n)*k*z(n)*Gamma(n)*fn(n)^2-LamdaA(n)*phi1(n);
end

function [O3_u,O3_q]=Cloud_processing(LamdaT,p_cloud,z,Gamma,Rc,fc,LamdaE,c_cloud,LamdaA,tn,n,phi3)
O3_u=LamdaT(n)*(z(n)/Rc(n)+tn+z(n)*Gamma(n)/fc)+LamdaE(n)*p_cloud*z(n)/Rc(n)-LamdaA(n)*phi3(n);
O3_q=c_cloud*z(n)*Gamma(n);
end

function allowed = update_allowed(M,bn,allowed,a,t,n)
for i = 1:M
    if ismember(i,bn) && allowed(i)==0
        if a(t,n)~=i
            allowed(i)=1;
        end
        allowed(M+1)=i;
        return
    end
end
end




