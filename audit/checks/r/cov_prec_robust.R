# cov_prec_robust external reference: rrcov::CovMcd / robustbase::covMcd (system R, R_LIBS_USER=~/R/audit-lib).
#
# Usage: Rscript cov_prec_robust.R in.csv out.csv MODE H [INIT.csv] [NSEEDS]
#   in.csv  : n x p matrix (no header), the exact array the sklearn fit saw (pyspi's z-scored input)
#   out.csv : long format "key,k,i,j,value" (k = seed index or 0; i, j 1-based; scalars use i = j = 0)
#   MODE    : "own" -> rrcov::CovMcd FAST-MCD with its own random search (nsamp = 500 default),
#                      one fit per R seed k = 1..NSEEDS (set.seed(k)), h matched to sklearn's h,
#                      use.correction = FALSE; plus one fit with rrcov defaults (alpha = 0.5, i.e. the
#                      canonical h = floor((n+p+1)/2), use.correction = TRUE) under set.seed(1).
#             "det" -> robustbase::covMcd(nsamp = "deterministic", initHsets = INIT) (rrcov::CovMcd does not
#                      forward initHsets), i.e. C-steps started from sklearn's raw support; no randomness.
#   H       : sklearn's raw h-subset size; alpha is chosen so that robustbase::h.alpha.n(alpha, n, p) == H.
#
# Conventions recorded (robustbase 0.99.2 covMcd source, deparsed in this R): raw.cov = .MCDcons(p, h/n) *
# cov(H) with divisor h - 1 (classPC eigenvalues / (n - 1) in .detmcd); reweighting weights
# 1[mah < qchisq(0.975, p)] from the raw centre/raw.cov; cov = cov.wt(x, w) (divisor sum(w) - 1) times
# .MCDcons(p, 0.975) ONLY when sum(w) != n; .MCDcons(p, a) = a / pchisq(qchisq(a, p), p + 2).
# Outputs both $raw.cov and the reweighted $cov, plus sum(raw.weights) so that the caller can map the
# divisor to sklearn's MLE convention.
suppressMessages({library(robustbase); library(rrcov)})
a <- commandArgs(trailingOnly = TRUE)
inf <- a[1]; outf <- a[2]; mode <- a[3]; H <- as.integer(a[4])
x <- unname(as.matrix(read.csv(inf, header = FALSE)))
n <- nrow(x); p <- ncol(x); n2 <- (n + p + 1) %/% 2
alpha <- (H - (2 * n2 - n) + 1e-9) / (2 * (n - n2))
stopifnot(h.alpha.n(alpha, n, p) == H)
rows <- list()
put <- function(key, k, M) {
  M <- as.matrix(M)
  idx <- expand.grid(i = seq_len(nrow(M)), j = seq_len(ncol(M)))
  rows[[length(rows) + 1]] <<- data.frame(key = key, k = k, i = idx$i, j = idx$j, value = as.vector(M))
}
put("n", 0, n); put("p", 0, p); put("h", 0, H); put("alpha", 0, alpha)
put("r_version", 0, as.numeric(paste(R.version$major, sub("\\.", "", R.version$minor), sep = ".")))
if (mode == "own") {
  nseeds <- if (length(a) >= 5) as.integer(a[5]) else 20L
  for (k in seq_len(nseeds)) {
    set.seed(k)
    f <- CovMcd(x, alpha = alpha, use.correction = FALSE)
    put("cov", k, getCov(f)); put("raw_cov", k, f@raw.cov); put("sum_w", k, sum(f@raw.wt))
    put("crit", k, f@crit); put("raw_cnp2", k, f@raw.cnp2); put("cnp2", k, f@cnp2)
  }
  set.seed(1)
  g <- CovMcd(x)   # rrcov defaults: alpha = 0.5 (h = floor((n+p+1)/2)), use.correction = TRUE
  put("default_cov", 0, getCov(g)); put("default_h", 0, g@quan); put("default_sum_w", 0, sum(g@raw.wt))
  put("default_raw_cnp2", 0, g@raw.cnp2); put("default_cnp2", 0, g@cnp2)
} else if (mode == "det") {
  init <- as.integer(scan(a[5], quiet = TRUE))
  f <- covMcd(x, alpha = alpha, nsamp = "deterministic", initHsets = matrix(init, ncol = 1),
              use.correction = FALSE)
  put("cov", 0, f$cov); put("raw_cov", 0, f$raw.cov); put("sum_w", 0, sum(f$raw.weights))
  put("raw_weights", 0, f$raw.weights); put("center", 0, f$center); put("raw_center", 0, f$raw.center)
  put("best", 0, sort(f$best)); put("n_csteps", 0, f$n.csteps); put("crit", 0, f$crit)
  put("raw_cnp2", 0, f$raw.cnp2); put("cnp2", 0, f$cnp2)
  put("best_equals_init", 0, as.numeric(identical(sort(as.integer(f$best)), sort(init))))
} else stop("unknown mode ", mode)
write.csv(do.call(rbind, rows), outf, row.names = FALSE)
cat("robustbase", as.character(packageVersion("robustbase")), "rrcov", as.character(packageVersion("rrcov")), "\n")
