clc;
clear;
close all;

% NMSE versus gamma_a
gamma_a_all = [1e4,5e4,1e5,5e5,1e6,5e6,1e7];
% gamma_a_all = [1e4,5e4];
SNRdB = 40; % [40]
N = 512; % signal dimension [1024]
del = 0.5; % measurement rate M/N [0.5]
rho = 0.2; % normalized sparsity rate E{K}/M=(E{K}/N)/del [0.2]
xmean1 = 0; % prior mean of non-zero x coefs
Afro2 = N;
M = round(del*N); %128
xvar1 = 1; 
wvar = (Afro2/M)*10^(-SNRdB/10)*rho*(abs(xmean1)^2+xvar1); 
iter_all = 50;
Theo_mse = zeros(iter_all,length(gamma_a_all));
mse_vamp_PI = zeros(iter_all,length(gamma_a_all));
mse_vamp_PC = zeros(iter_all,length(gamma_a_all));
K_iter = 50;
for gamma_index = 1:length(gamma_a_all)
    for iter = 1:iter_all
    % generate signal 
        x = zeros(N,1);
        supp = find(rand(N,1)<rho); 
        K = length(supp);
        x(supp,1) = xmean1 + sqrt(xvar1)*randn(K,1);
        % generate noise 
        w = sqrt(wvar)*randn(M,1);
        A = randn(M,N);
        R = rank(A);
        A = sqrt(Afro2/trace(A'*A))*A;
        % generate perturbation 
        gamma_a = gamma_a_all(gamma_index);
        Q = N;
        perb_add = zeros(size(A));
        a = sqrt(1/gamma_a)*randn(Q,1);
        A_perb = zeros(M,N,Q);
        for i = 1:Q
            A_perb(:,:,i) = randn(M,N);
            A_perb(:,:,i) = sqrt(Afro2/trace(A_perb(:,:,i)'*A_perb(:,:,i)))*A_perb(:,:,i);
            perb_add = perb_add+a(i)*A_perb(:,:,i);
        end
        y = (A+perb_add)*x+w;
        A_supp = A(:,supp);
     % true noise covariance
        Tau_true = zeros(M,M);
        for i = 1:Q
            Tau_true = Tau_true+1/gamma_a*A_perb(:,:,i)*x*x'*A_perb(:,:,i)';
        end
        Tau_true = Tau_true+wvar*eye(M);
        x_LMMSE = xvar1*A_supp'/(Tau_true+xvar1*(A_supp*A_supp'))*y;
        x_ls = zeros(size(x));
        x_ls(supp) = x_LMMSE;
        Theo_mse(iter,gamma_index) = 20*log10(norm(x_ls(:,1)-x(:,1))/norm(x(:,1)));
        gamma0 = 1e-7;
        gamma0_inv = 1/gamma0;
        [U_A,S_A,V_A]=svd(A,'econ');
        s_bar = diag(S_A);
        y_tilde = (diag(s_bar))\(U_A')*y;
        d_k = (1/wvar)*((diag(1/wvar*s_bar.^2+eps))\(s_bar.^2));
        r0 = N/R*V_A*(diag(d_k/mean(d_k)))*y_tilde;
        gamma_kold = gamma0;
        gamma_min = 1e-11;
        gamma_max = 1e11;
        mse_vamp = zeros(K_iter,1);
        mse_PI = zeros(K_iter,1);
        for i = 1:K_iter
            L = 0.5*log(gamma0_inv./(gamma0_inv+xvar1))+r0.^2/2*gamma0-(r0-xmean1).^2/2/(gamma0_inv+xvar1);
            post_mean= (xvar1*r0+gamma0_inv*xmean1)/(gamma0_inv+xvar1);
            post_prior=rho./(rho+(1-rho)*exp(-L));
            post_var = xvar1.*gamma0_inv./(gamma0_inv+xvar1);
            x_hat = post_prior.*post_mean;
            mse_vamp(i) = 20*log10(norm(x_hat(:)-x(:))/norm(x(:)));
            alpha_k = mean(post_prior.*(post_var+post_mean.^2)-x_hat.^2)*gamma0;     %  A_1
            r_tilde = (x_hat-alpha_k.*r0)./(1-alpha_k);      %r_2k
            gamma_tilde = gamma0.*(1-alpha_k)./alpha_k;      %gamma_2k
         % obtain  d_k,gamma_{k+1},r_{k+1}
            Cal_Tau2k = zeros(M,M);
            Cov_appx = r_tilde*r_tilde'+(1/gamma_tilde+eps)*eye(N);
            for i_q = 1:Q
                Cal_Tau2k = Cal_Tau2k+1/gamma_a*A_perb(:,:,i_q)*Cov_appx*A_perb(:,:,i_q)';
            end
            Cal_Tau2k = Cal_Tau2k+wvar*eye(M);
            Cal_Tau2k_sqrt = sqrtm(Cal_Tau2k);
            A_eq_2k = Cal_Tau2k_sqrt\A;
            gamma_w_2k = norm(A_eq_2k,'fro')^2/N;
            A_eq_2k = A_eq_2k/sqrt(gamma_w_2k);
            y_eq_2k = Cal_Tau2k_sqrt\y/sqrt(gamma_w_2k);
            wvar_eq_2k = 1/gamma_w_2k;    
            [U_A_eq,S_A_eq,V_A_eq]=svd(A_eq_2k,'econ');
            s_bar_eq = diag(S_A_eq);
            y_tilde_eq = (diag(s_bar_eq))\(U_A_eq')*y_eq_2k;  
            d_k_eq = (1/wvar_eq_2k)*((diag(1/wvar_eq_2k*s_bar_eq.^2+gamma_tilde))\(s_bar_eq.^2));
            gamma_kplus1 = gamma_tilde*R*mean(d_k_eq)/(N-R*mean(d_k_eq));
            r_kplus1 = r_tilde+N/R*V_A_eq*(diag(d_k_eq/mean(d_k_eq)))*(y_tilde_eq-V_A_eq'*r_tilde);
            r0 = r_kplus1;
            gamma0 = gamma_kplus1;
            gamma0 = min(gamma0,gamma_max);
            gamma0 = max(gamma0,gamma_min);
            gamma0_inv = 1/gamma_kplus1; 
        end
        
      %% Standard VAMP
        gamma0 = 1e-7;
        gamma0_inv = 1/gamma0;
        y_tilde = (diag(s_bar))\(U_A')*y;
        d_k = (1/wvar)*((diag(1/wvar*s_bar.^2+eps))\(s_bar.^2));
        r0 = N/R*V_A*(diag(d_k/mean(d_k)))*y_tilde;

        gamw = 1/wvar;
        d = s_bar.^2;
        lam = gamw*d; % only contains nonzero entries (usually M of them)
        lam_ = [lam;zeros(N-length(lam),1)]; % all N entries

        for i = 1:K_iter
            L = 0.5*log(gamma0_inv./(gamma0_inv+xvar1))+r0.^2/2*gamma0-(r0-xmean1).^2/2/(gamma0_inv+xvar1);
            post_mean= (xvar1*r0+gamma0_inv*xmean1)/(gamma0_inv+xvar1);
            post_prior=rho./(rho+(1-rho)*exp(-L));
            post_var = xvar1.*gamma0_inv./(gamma0_inv+xvar1);
            x_hat = post_prior.*post_mean;
            mse_PI(i) = 20*log10(norm(x_hat(:)-x(:))/norm(x(:)));

            alpha_k = mean(post_prior.*(post_var+post_mean.^2)-x_hat.^2)*gamma0;     %  A_1
            r_tilde = (x_hat-alpha_k.*r0)./(1-alpha_k);      %r_2k
            gamma_tilde = gamma0.*(1-alpha_k)./alpha_k;      %gamma_2k
            d_k = (1/wvar)*((diag(1/wvar*s_bar.^2+gamma_tilde))\(s_bar.^2));

            gamma_kplus1 = gamma_tilde*R*mean(d_k)/(N-R*mean(d_k));
            r_kplus1 = r_tilde+N/R*V_A*(diag(d_k/mean(d_k)))*(y_tilde-V_A'*r_tilde);

            r0 = r_kplus1;
            gamma0 = gamma_kplus1;
            gamma0 = min(gamma0,gamma_max);
            gamma0 = max(gamma0,gamma_min);
            gamma0_inv = 1/gamma_kplus1; 
        end
        
        
%         [mse_vamp, mse_ls_perbignored, x_LMMSE_perbignored, x_hat_vamp ] ......
%     = VAMPsolve(A, y, supp, x, wvar, N, del, rho, dampfac);
       mse_vamp_PC(iter,gamma_index) = mse_vamp(end);
       mse_vamp_PI(iter,gamma_index) = mse_PI(end);
    end
end

figure(2)
plot(log10(gamma_a_all),median(mse_vamp_PC),'-ro')
hold on
plot(log10(gamma_a_all),median(Theo_mse),'-.k')
hold on 
plot(log10(gamma_a_all),median(mse_vamp_PI),'-.b*')
hold off
legend('PC-VAMP','oracle LMMSE','PI-VAMP')
xlabel('log \gamma_a')
ylabel('median NMSE(dB)')
figure(3)
plot(log10(gamma_a_all),mean(mse_vamp_PC),'-ro')
hold on
plot(log10(gamma_a_all),mean(Theo_mse),'-.k')
hold on 
plot(log10(gamma_a_all),mean(mse_vamp_PI),'-.b*')
hold off
legend('PC-VAMP','oracle LMMSE','PI-VAMP')
xlabel('log \gamma_a')
ylabel('mean NMSE(dB)')

save('NMSEversusgammaA.mat')
