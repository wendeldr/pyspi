# dcorr_hsic external cross-check: R package dHSIC (Pfister, Buehlmann, Schoelkopf & Peters 2018,
# JRSS-B 80:5-31), dhsic() = the biased V-statistic HSIC_b = (1/n^2) tr(K H L H) for d = 2 variables
# (dhsic body: term1/n^2 + term2 - term3 with term2 = prod(sum K)/n^4, term3 = 2/n^3 sum(colSums K * colSums L)).
# Invoked by refs/dcorr_hsic.py:  Rscript r/dcorr_hsic.R <exchange_dir>
# <exchange_dir>/jobs.csv: columns job (int); in_<job>.csv: T x P matrix (no header, 17 significant digits).
# out_<job>.csv: one row per ORDERED pair (i, j), i != j (1-based), x = column i, y = column j, columns:
#   i, j,
#   sigma_x       median of the off-diagonal |x_a - x_b| computed with stats::dist + stats::median (the hyppo
#                 median-heuristic convention: gamma = 1/(2 sigma^2) from x ONLY, reused for y)
#   hb_xy, hb_xx, hb_yy   dhsic(.., kernel = "gaussian.fixed", bandwidth = sigma_x) for (x,y), (x,x), (y,y);
#                 dHSIC's gaussian_grammat_rcpp(x, bw) = exp(-|x_a - x_b|^2 / (2 bw^2)) (checked on a 3-point
#                 example), so bandwidth = sigma_x reproduces hyppo's exp(-gamma d^2)
#   hu_xy, hu_xx, hu_yy   unbiased HSIC (Song et al. 2012 JMLR 13:1393, eq. 5) written here in R on the dHSIC
#                 Gram matrices (gaussian_grammat_rcpp, bandwidth sigma_x) with zeroed diagonals; this part is an
#                 R re-implementation, not a dHSIC function (dHSIC has no unbiased estimator)
#   def_xy, def_xx, def_yy, bw_x, bw_y   dhsic() with its own defaults (kernel = "gaussian": a SEPARATE median
#                 bandwidth per variable, bw = sqrt(median(d^2)/2) from at most 1000 randomly chosen rows;
#                 set.seed(42) before every call so the subsample is reproducible)
suppressPackageStartupMessages(library(dHSIC))
args <- commandArgs(trailingOnly = TRUE)
ex <- args[1]
jobs <- read.csv(file.path(ex, "jobs.csv"), stringsAsFactors = FALSE)

gram <- function(v, bw) dHSIC:::gaussian_grammat_rcpp(matrix(v, ncol = 1), bw, length(v), 1L)

hsic_u <- function(K, L) {
  n <- nrow(K)
  diag(K) <- 0; diag(L) <- 0
  kr <- rowSums(K); lr <- rowSums(L)
  (sum(K * L) + sum(K) * sum(L) / ((n - 1) * (n - 2)) - 2 / (n - 2) * sum(kr * lr)) / (n * (n - 3))
}

fx <- function(v) sprintf("%.17g", v)
for (k in seq_len(nrow(jobs))) {
  X <- as.matrix(read.csv(file.path(ex, paste0("in_", jobs$job[k], ".csv")), header = FALSE))
  storage.mode(X) <- "double"
  P <- ncol(X)
  rows <- list()
  for (i in seq_len(P)) {
    x <- X[, i]
    sx <- stats::median(as.vector(stats::dist(x)))
    Kx <- gram(x, sx)
    for (j in seq_len(P)) {
      if (i == j) next
      y <- X[, j]
      Ly <- gram(y, sx)
      fixed <- function(a, b) dhsic(a, b, kernel = c("gaussian.fixed", "gaussian.fixed"), bandwidth = c(sx, sx))$dHSIC
      hb_xy <- fixed(x, y); hb_xx <- fixed(x, x); hb_yy <- fixed(y, y)
      hu_xy <- hsic_u(Kx, Ly); hu_xx <- hsic_u(Kx, Kx); hu_yy <- hsic_u(Ly, Ly)
      set.seed(42); d_xy <- dhsic(x, y)
      set.seed(42); d_xx <- dhsic(x, x)$dHSIC
      set.seed(42); d_yy <- dhsic(y, y)$dHSIC
      rows[[length(rows) + 1]] <- c(i, j, fx(sx), fx(hb_xy), fx(hb_xx), fx(hb_yy), fx(hu_xy), fx(hu_xx), fx(hu_yy),
                                    fx(d_xy$dHSIC), fx(d_xx), fx(d_yy), fx(d_xy$bandwidth[1]), fx(d_xy$bandwidth[2]))
    }
  }
  out <- do.call(rbind, rows)
  colnames(out) <- c("i", "j", "sigma_x", "hb_xy", "hb_xx", "hb_yy", "hu_xy", "hu_xx", "hu_yy",
                     "def_xy", "def_xx", "def_yy", "bw_x", "bw_y")
  write.csv(out, file.path(ex, paste0("out_", jobs$job[k], ".csv")), row.names = FALSE, quote = FALSE)
}
cat(sprintf("dcorr_hsic.R: %d jobs, R %s, dHSIC %s\n", nrow(jobs), paste(R.version$major, R.version$minor, sep = "."),
            as.character(utils::packageVersion("dHSIC"))))
