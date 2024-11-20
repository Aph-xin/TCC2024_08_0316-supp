function [Eta, gamma_k] = master_solver(N,UB, L, dL, bo, b_range, b_tra, isFmin)

    if isFmin == 1 
        eps_rnd = 2;
        % Variable %
        syms eta;
        syms bv [1 N];
        ksi = L+dL.*(bv-bo)-eta;
        
        % ================== Resolve Optimality cuts =================== %
        options = optimoptions(@fmincon, 'Display', 'off');
        disp('fmincon is solving beta, please wait ...');
        % Object Function %
        func = @(x)x(1);
        % Random Initialization % 
        x0 = [-996, max(b_range)*ones(1,N)];
        % Linear Inequality Eqs %
        A = []; b = [];
        % Linear equality Eqs %
        Aeq = []; beq = [];
        % Variable Constrains %
        lb = [-Inf, min(b_range)*ones(1,N)]; 
        ub = [Inf, max(b_range)*ones(1,N)];
        % Nonlinear Constrain %
        c = @(x)double(subs(subs(ksi,eta,x(1)),bv,x(2:N+1)));
        % Special Condition for beta Only %
        ceq = @(x)round(x(2:N+1))-round(x(2:N+1),eps_rnd);
        nonlcon = @(x)deal(c(x),ceq(x));
        % Solution %
        x = fmincon(func,x0,A,b,Aeq,beq,lb,ub,nonlcon,options);
        LB = x(1); Eta = x(1); 
        %gamma_k = x(2:N+1);
        gamma_k = round(x(2:N+1),eps_rnd);
        
        % Display: %
        disp(['Current Solution for beta: ', num2str(gamma_k)]);
        disp(['Current MasterP minEta: ', num2str(Eta)]);
    
    else
        % 遍历压缩
        disp('Traversing all the compression ratio, please wait ...');
        base = b_tra; % 可选的元素
        len = length(base); % 总的排列数量
        beta = zeros(len^N,N);
        % 生成所有的排列
        for i = 0:len^N-1
            % 使用dec2base将i转换为3进制数，然后将每个字符转换为数字，最后转换为索引
            indices = double(dec2base(i, len, N)) - '0' + 1;
            % 使用索引从base中选择元素
            beta(i+1, :) = base(indices);
        end
        beta=beta/100;
        Cost=zeros(1,len^N);
        
        for i=1:len^N
            x=beta(i,:);
            Cost(i)= max(L+dL.*(x-bo));
        end
    
        [Eta,ay]=min(Cost);%min,index
        gamma_k=beta(ay,:);
    
        % Display: %
        disp(['Current Solution for beta: ', num2str(gamma_k)]);
        disp(['Current MasterP minEta: ', num2str(Eta)]);  
    end
end
