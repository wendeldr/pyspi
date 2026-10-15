1;
% gwtau family, SAME-LINEAGE reference (GNU Octave): the authors' MATLAB algorithm for GW-tau,
% transcribed loop for loop from github.com/kravtsova2/GWtau (GWtau_tutorial/vec_geo_dist.m,
% wass_sorted.m, read 2026-10-15; trajectory construction from gwtau_tutorial_main.m:
% time = (0:n-1)', Traj = [time X]).  pyspi/statistics/distance.py:273-315 is a Python port of these
% files, so this reference is same-lineage: it checks the port, not the definition.
%
% Usage: octave-cli --no-gui -q gwtau.m <job_dir>
%   <job_dir>/rows.csv  P x T matrix (one series per row, %.17g)
%   <job_dir>/spec.csv  K x 4 rows (i, j, n1, n2), 1-based series indices, lengths n1, n2
%   writes <job_dir>/result.csv  K values: wass_sorted(vec_geo_dist([t1 x_i(1:n1)]), vec_geo_dist([t2 x_j(1:n2)]))

function v = vec_geo_dist(X)
  % loop i = 1..n-1: v(i) = dold + norm(X(i,:) - X(i+1,:)); the root (distance 0) is not stored
  dold = 0;
  for i = 1:size(X, 1) - 1
    dnew = norm(X(i, :) - X(i + 1, :));
    v(i) = dold + dnew;
    dold = v(i);
  end
end

function res = wass_sorted(v1, v2)
  v1 = sort(v1, 'descend');
  v2 = sort(v2, 'descend');
  if length(v1) == length(v2)
    res = sum((v1 - v2) .^ 2) / length(v1);
    res = sqrt(res);
  else
    N = length(v1);
    M = length(v2);
    my_sum = 0;
    for i = 1:N
      for j = 1:M
        % integer cross-multiplied overlap test, as in the authors' file
        if min(i * M, j * N) > max((i - 1) * M, (j - 1) * N)
          lam = min(i / N, j / M) - max((i - 1) / N, (j - 1) / M);
          my_sum = my_sum + lam * (v1(i) - v2(j)) ^ 2;
        end
      end
    end
    res = sqrt(my_sum);
  end
end

args = argv();
job = args{1};
X = dlmread(fullfile(job, 'rows.csv'), ',');
S = dlmread(fullfile(job, 'spec.csv'), ',');
out = zeros(size(S, 1), 1);
for k = 1:size(S, 1)
  i = S(k, 1); j = S(k, 2); n1 = S(k, 3); n2 = S(k, 4);
  xi = X(i, 1:n1)'; xj = X(j, 1:n2)';
  ti = (0:n1 - 1)'; tj = (0:n2 - 1)';
  out(k) = wass_sorted(vec_geo_dist([ti xi]), vec_geo_dist([tj xj]));
end
fid = fopen(fullfile(job, 'result.csv'), 'w');
fprintf(fid, '%.17g\n', out);
fclose(fid);
