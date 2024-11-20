function [Eta, x_k, gamma_k] = primalP(V, Q, ...      
        N, M, c, Rc, Rm, tn, p_cloud, p_fog, fn, fc, fm, k, c_cloud, c_fog, ...
        alpha, LamdaT, LamdaE, LamdaA, z_k, Gamma, H ,va, theta)
    % 遍历压缩
    a_x = [];
    base = [8 9]; % 可选的元素
    len = length(base); % 总的排列数量
    beta = zeros(len^N,N);
    % 生成所有的排列
    for i = 0:len^N-1
        % 使用dec2base将i转换为3进制数，然后将每个字符转换为数字，最后转换为索引
        indices = double(dec2base(i, len, N)) - '0' + 1;
        % 使用索引从base中选择元素
        beta(i+1, :) = base(indices);
    end
    beta=beta/10;
    %disp(beta)
    pen=zeros(1,N);
    penalty=zeros(1,len^N);
    Cost=zeros(1,len^N);
    Oc=zeros(1,N);
    Oc_q=zeros(1,N);
    n_m = zeros(1, M);

    for i=1:len^N
        x=beta(i,:);
        phi1=0; phi2=0; phi3=0;
        phi1=va(1,:).*log2(theta(1,:).*x);
        phi2=va(2,:).*log2(theta(2,:).*x);
        phi3=va(3,:).*log2(theta(3,:).*x);
        z=z_k.*x;
        a = CATS_algorithm(V, Q, x, ...      
        N, M, c, Rc, Rm, tn, p_cloud, p_fog, fn, fc, fm, k, c_cloud, c_fog, ...
        alpha, LamdaT, LamdaE, LamdaA, z, Gamma, H ,phi1, phi2, phi3);    
        a_x = [a_x;a];
        for n=1:N
            if a(n)==0
                pen(n)=LamdaT(n)*z(n)*Gamma(n)/fn(n)+LamdaE(n)*k*z(n)*Gamma(n)*fn(n)^2-LamdaA(n)*phi1(n);
                %Local_processing(LamdaT,LamdaE,LamdaA,k,z,Gamma,fn,n,x, phi1(i,:));
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
        penalty(i)=sum(pen);
        Cost(i)=sum(Oc);
    end

    [Eta,ay]=min(Cost);%index
    gamma_k=beta(ay,:);%压缩率
    x_k = a_x(ay,:); %最佳决策

    % Display: %
    disp(['Current x: ', num2str(x_k)]);
    disp(['Current Solution for beta: ', num2str(gamma_k)]);
    disp(['Current PrimalP minObj: ', num2str(Eta)]);  

end

function [O2_u,O2_q]=Fog_proccessing(LamdaT,LamdaE,LamdaA,z,Gamma,Rm,c_fog,fm,p_fog,alpha,a,n,t,p,n_m,M,x,phi2)
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
    O3_u=LamdaT(n)*(z(n)/Rc(n)+tn+z(n)*Gamma(n)/fc)+LamdaE(n)*p_cloud*z(n)/Rc(n)-LamdaA(n)*phi3(n); %%%%%
    O3_q=c_cloud*z(n)*Gamma(n);
end