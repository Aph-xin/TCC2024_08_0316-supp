function a_n = CATS_algorithm(V, Q, gamma, ...      
        N, M, c, Rc, Rm, tn, p_cloud, p_fog, fn, fc, fm, k, c_cloud, c_fog, ...
        alpha, LamdaT, LamdaE, LamdaA, z_k, Gamma, H ,phi1,phi2,phi3)
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
    
    a_n = a(T,:); % 决策
    
    %disp("==================================================");
    %disp(['Final Solution to (x*, y*): (',num2str(x),',',...
    %    num2str(gamma_k),')']);
    % disp(['Final Solution the problem: ', num2str(UB)]); 

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

