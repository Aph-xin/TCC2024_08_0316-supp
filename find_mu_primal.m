function [mu] = find_mu_primal(N,M,x_val, b_, relaxed_param, opt_func, ...
    cst_func) % Solve μ or λ %

    syms xv [M+2 N] 
    syms bv [1 N]
    % Take care of Precision Error %
    % eps_rnd = -1; % You're ALLOWED to EDIT eps_rnd %
    g = subs(subs(cst_func-relaxed_param, xv, x_val),bv,b_);

    if (relaxed_param == 0)
        syms x [M+2 N]% 未知数
        syms bv [1 N]
        syms mu [1 (M+2)*N]
        gx = subs(cst_func, xv, x);%1
        P  = subs(opt_func,bv,b_);
        P  = subs(P, xv, x);%1
        L  = P + gx*mu;
        % Building equations to solve Lagrange Multipliers %
        % Eq = roundn(double(g), eps_rnd) == 0;
        for i=1:M+2
            for n=1:N
                Eq((i-1)*N+n) = subs(diff(L((i-1)*N+n),x(i,n)),x(i,n),x_val(i,n)) == 0;
            end
        end
    else
        syms s mu
        gx = subs(subs(cst_func, xv, x_val), bv, b_);
        P  = s;
        L  = P + (gx-s)*mu;
        % Building equations to solve Lagrange Multipliers %
        % Eq = roundn(double(g), eps_rnd) == 0;
        Eq(1:N) = subs(diff(L,s), s, relaxed_param) == 0;
    end
    
    disp("vpasolve calculating, please wait ..."); 
    % Solve the system of equations and display the results %
    mu = vpasolve(Eq, mu); 
    % Matlab Can't Solve? Please increase eps_rnd. %
    mu = double(struct2array(mu));
    mu = reshape(mu, M+2, N);
    mu = zeros(M+2,N);
    % Show results in a vector of data type double %
    if (relaxed_param == 0)
        disp("Current mu^T: "); 
    else
        disp("Current lambda^T: "); 
    end
    disp(mu(1)); 
end