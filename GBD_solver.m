function [sol, UB, LB, a_n, gamma_k] = GBD_solver(V, Qe, beta, ...      
        N, M, c, Rc, Rm, tn, p_cloud, p_fog, fn, fc, fm, k_p, c_cloud, c_fog, ...
        alpha, LamdaT, LamdaE, LamdaA, z, Gamma, round, H ,va, theta, ...
        b_range, b_tra, isFmin)
    
    % step-1: init
    % parameter init
    K = round;
    gamma = [];
    gamma = [gamma;beta];
    ksi = []; ksi_f=[];
    k = 1; eps = 1;
    K_feas = 0; K_infeas = 0;
    mu = []; lambda = [];
    UB = +inf; LB = -inf;

    while abs(UB-LB) > eps && k < K+1
        disp("==================================================");
        disp(['Epoch: ', num2str(k)]);
        % Iteration Update
        k = k + 1;
        phi1=va(1,:).*log2(theta(1,:).*gamma(end,:));
        phi2=va(2,:).*log2(theta(2,:).*gamma(end,:));
        phi3=va(3,:).*log2(theta(3,:).*gamma(end,:));
        % step-2: solve primal problem
        [isExist, mu_k, L_k, dL_k, UB_k, ~, a_n] = primal_solver(V, Qe, gamma(end,:), ...      
        N, M, c, Rc, Rm, tn, p_cloud, p_fog, fn, fc, fm, k_p, c_cloud, c_fog, alpha, ...
        LamdaT, LamdaE, LamdaA, z, Gamma, H ,va,theta, phi1, phi2, phi3);
        if isExist == 1
            % update upper bound
            UB = min(UB, UB_k);
            % feasibility count
            K_feas = K_feas + 1;
            %μ: lagrange Multiplier
            mu = [mu; mu_k];
%         else
%             % solve relaxed problem
%             [ksi_k] = relaxed_solver(gamma_k,K,J);
%             % infeasibility count
%             K_infeas = K_infeas + 1;
%             %λ: Lagrange Multiplier
%             lambda = [lambda;lambda_k];
%             %ξ: Add Feas Cut
%             ksi_f = [ksi_f; ksi_k];
        end

        % step-3: solve master problem
        [LB_k, gamma_k] = master_solver( N, UB, L_k, dL_k, gamma(end,:), ...
            b_range, b_tra, isFmin);
        % update lower bound
        % LB = max(LB, LB_k);
        LB = LB_k;
        %ξ: add Opt Cut
        % ksi = [ksi; ksi_k]; % a_n
        disp(['Current Upper Bound UB: ', num2str(UB)]);
        disp(['Current Lower Bound LB: ', num2str(LB)]);
        %beta: Solution to MP
        gamma = [gamma; gamma_k];
    end
    sol = [gamma_k, a_n];% beta, x
    disp("==================================================");
    %disp(['Final Solution to (x*, y*): (',num2str(x),',',...
    %    num2str(gamma_k),')']);
    disp(['Final Solution the problem: ', num2str(UB)]); 

end