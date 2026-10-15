1;  % script file (not a function file): local functions follow
% Octave 8.4 reference for the phi family (audit/checks/refs/phi.py invokes it).
%
% Runs pyspi's vendored PhiToolbox copy (git show 76ab2bb^:pyspi/lib/PhiToolbox, extracted to
% the session scratchpad; Phi/ and utility/ and utility/Gauss/ are addpath()ed) on the inputs that
% refs/phi.py writes under out/exchange/phi/.
%
% Usage: octave-cli --no-gui -q phi.m <toolbox_root> <jobs.csv> <out.csv>
%   jobs.csv rows: job_id, kind, phitype, normalization, tau
%     kind 1: data job, file job_<id>.csv holds X (2 x T), computed through phi_comp(X, Z, params, options)
%             with Z = [1 2], params.tau = tau, options.type_of_dist = 'Gauss',
%             options.type_of_phi = 'star'|'Geo', options.normalization = 0|1
%             (the struct the pre-eb9d471 pyspi wrapper passed through oct2py; tau was hard-coded 1)
%     kind 2: covariance job, file job_<id>.csv holds [Cov_X; Cov_XY; Cov_Y] (6 x 2); computed through
%             phi_Gauss(type_of_phi, Z, probs, options) (the function phi_comp dispatches to)
%     phitype 1 = 'star', 2 = 'Geo'
%   out.csv rows: job_id, phi, I, beta_fd, phi_gradobj, beta_gradobj, offdiag_Ap
%     phi          : the vendored path (phi_comp / phi_Gauss); star uses fminunc with finite differences
%     I, beta_fd   : [phi, I, beta] from phi_star_Gauss on the same covariances (star only)
%     phi_gradobj  : star only: the same setup but fminunc with optimset('GradObj','on'), i.e. the analytic
%                    gradient of utility/I_s_I_s_d.m consumed (the published toolbox calls minFunc with that
%                    gradient; minFunc itself is not vendored, so this is the closest runnable analogue)
%     offdiag_Ap   : Geo only: max |off-diagonal A_p| returned by phi_G_Gauss (AL constraint residual)
args = argv();
root = args{1};
addpath(fullfile(root, 'Phi'));
addpath(fullfile(root, 'utility'));
addpath(fullfile(root, 'utility', 'Gauss'));
jobs = dlmread(args{2}, ',');
outf = args{3};
exch = fileparts(args{2});
warning('off', 'all');

function [phi_star, beta_opt] = phi_star_gradobj(Cov_X, Cov_XY, Cov_Y, Z, normalization)
  % Copy of phi_star_Gauss.m lines 50-126 (vendored) with the optimiser call changed to consume
  % the analytic gradient returned by I_s_I_s_d (second output).
  N = size(Cov_X,1);
  Cov_Y_X = Cov_cond(Cov_Y,Cov_XY',Cov_X);
  H_cond = H_gauss(Cov_Y_X);
  H = H_gauss(Cov_Y);
  I = H - H_cond;
  N_c = max(Z);
  M_cell = cell(N_c,1);
  for i=1:N_c
    M_cell{i} = find(Z==i);
  end
  X_D = zeros(N,N); YX_D = zeros(N,N); C_D_cond = zeros(N,N);
  for i=1:N_c
    M = M_cell{i};
    Cov_X_p = Cov_X(M,M); Cov_Y_p = Cov_Y(M,M); Cov_YX_p = Cov_XY(M,M)';
    X_D(M,M) = Cov_X_p; YX_D(M,M) = Cov_YX_p; C_D_cond(M,M) = Cov_cond(Cov_Y_p,Cov_YX_p,Cov_X_p);
  end
  Cov_X_inv = inv(Cov_X);
  C_D_beta1_inv = X_D\YX_D'/C_D_cond*YX_D/X_D;
  S_left = C_D_cond'\YX_D/X_D;
  S_right = X_D\YX_D'/C_D_cond;
  I_s_d_Const_part = 1/2*N;
  fh = @(beta)I_s_I_s_d(beta,C_D_beta1_inv,Cov_X_inv,Cov_X,Cov_Y,C_D_cond,S_left,S_right,I_s_d_Const_part);
  [beta_opt, minus_I_s] = fminunc(fh, 1, optimset('GradObj', 'on'));
  phi_star = I + minus_I_s;
  if normalization == 1
    H_p = zeros(N_c,1);
    for i=1:N_c
      M = M_cell{i};
      H_p(i) = H_gauss(Cov_X(M,M));
    end
    if N_c == 1
      phi_star = phi_star/H_p(1);
    else
      phi_star = phi_star/((N_c-1)*min(H_p));
    end
  end
end

out = nan(size(jobs,1), 7);
Z = [1 2];
for r = 1:size(jobs,1)
  jid = jobs(r,1); kind = jobs(r,2); pt = jobs(r,3); nrm = jobs(r,4); tau = jobs(r,5);
  M = dlmread(fullfile(exch, sprintf('job_%d.csv', jid)), ',');
  options = struct();
  options.type_of_dist = 'Gauss';
  if pt == 1
    options.type_of_phi = 'star';
  else
    options.type_of_phi = 'Geo';
  end
  options.normalization = nrm;
  params = struct();
  params.tau = tau;
  out(r,1) = jid;
  try
    if kind == 1
      phi = phi_comp(M, Z, params, options);
      probs = data_to_probs(M, params, options);
    else
      probs = struct('Cov_X', M(1:2,:), 'Cov_XY', M(3:4,:), 'Cov_Y', M(5:6,:));
      phi = phi_Gauss(options.type_of_phi, Z, probs, options);
    end
    out(r,2) = real(phi);
    if pt == 1
      [ps, I, b] = phi_star_Gauss(probs.Cov_X, probs.Cov_XY, probs.Cov_Y, Z, 1, nrm);
      out(r,3) = real(I); out(r,4) = b;
      [pg, bg] = phi_star_gradobj(probs.Cov_X, probs.Cov_XY, probs.Cov_Y, Z, nrm);
      out(r,5) = real(pg); out(r,6) = bg;
    else
      [pG, CEp, Ap] = phi_G_Gauss(probs.Cov_X, probs.Cov_XY, probs.Cov_Y, Z, [], nrm);
      out(r,7) = max(abs([Ap(1,2), Ap(2,1)]));
    end
  catch err
    fprintf(stderr, 'job %d failed: %s\n', jid, err.message);
  end
end
fid = fopen(outf, 'w');
for r = 1:size(out,1)
  fprintf(fid, '%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n', out(r,1), out(r,2), out(r,3), out(r,4), out(r,5), out(r,6), out(r,7));
end
fclose(fid);
fprintf('octave %s: %d jobs\n', version(), size(out,1));
