# External references for the coint family (R urca), called by refs/coint.py through Rscript.
#
# Usage: Rscript coint.R jo <in.csv> <out.csv>
#        Rscript coint.R df <in.csv> <out.csv>
# <in.csv>: T rows x P columns, no header (the array that reached pyspi's estimator).
#
# jo: urca::ca.jo on every unordered pair a < b with columns [x_b, x_a] (pyspi's first-visited
#     orientation: higher index first, order "hilo") and, for ecdet "none" / spec "transitory" only,
#     also [x_a, x_b] (order "lohi", column-order invariance).
#     K = k_ar_diff + 1 in {2, 11} (ca.jo's K is the VAR order in levels); type "eigen" and "trace";
#     the r = 0 statistic is the LAST element of @teststat (urca lists r <= P-1 first, r = 0 last);
#     eigenvalues from @lambda.
#     variants (hilo): ecdet none / const / trend, each with spec transitory and longrun;
#        none_dt = ecdet "none" on levels linearly detrended by OLS on [1, t] over the full sample
#                  (the pre-detrending statsmodels applies for det_order = 1, vecm.py:650-653, :670);
#        none_ut = ecdet "none" with dumvar = t (unrestricted linear trend in the VECM, Johansen case 5).
#     output columns: i,j,order,variant,spec,K,type,stat,lam1,lam2   (i, j 0-based, i = higher index)
# df: urca::ur.df on residuals(lm(y0 ~ y1)) (trend c) or residuals(lm(y0 ~ y1 + t)) (trend ct),
#     t = 1..T, type = "none" (no deterministic term in the ADF regression), for both orientations of
#     every pair: fixed lags p = 0..10 (selectlags "Fixed": sample = the T - 1 - p usable rows, i.e.
#     the largest sample for that p) and lags = 10 with selectlags "AIC" and "BIC" (urca's own search:
#     lags 1..10 only, on the common sample, no re-fit). Selected lag = number of lagged-difference
#     coefficients in @testreg.
#     output columns: dep,reg,trend,sel,p,tau   (dep, reg 0-based)
suppressPackageStartupMessages(library(urca))
args <- commandArgs(trailingOnly = TRUE)
mode <- args[1]; fin <- args[2]; fout <- args[3]
X <- as.matrix(read.csv(fin, header = FALSE))
N <- nrow(X); P <- ncol(X)
tt <- seq_len(N)
rows <- list()
add <- function(...) rows[[length(rows) + 1]] <<- data.frame(..., stringsAsFactors = FALSE)
r0 <- function(fit) { ts <- fit@teststat; ts[length(ts)] }

if (mode == "jo") {
  dtr <- function(v) residuals(lm(v ~ tt))
  for (a in 1:(P - 1)) for (b in (a + 1):P) {
    for (ord in c("hilo", "lohi")) {
      x <- if (ord == "hilo") cbind(X[, b], X[, a]) else cbind(X[, a], X[, b])
      colnames(x) <- c("y1", "y2")
      for (K in c(2L, 11L)) {
        vars <- if (ord == "hilo") c("none", "const", "trend", "none_dt", "none_ut") else c("none")
        specs <- if (ord == "hilo") c("transitory", "longrun") else c("transitory")
        for (v in vars) for (sp in specs) {
          if (v %in% c("none_dt", "none_ut") && sp == "longrun") next
          for (ty in c("eigen", "trace")) {
            fit <- if (v == "none_dt") {
              xd <- apply(x, 2, dtr); colnames(xd) <- colnames(x)
              ca.jo(xd, type = ty, ecdet = "none", K = K, spec = sp)
            } else if (v == "none_ut") {
              dv <- matrix(tt, ncol = 1); colnames(dv) <- "trend"
              ca.jo(x, type = ty, ecdet = "none", K = K, spec = sp, dumvar = dv)
            } else {
              ca.jo(x, type = ty, ecdet = v, K = K, spec = sp)
            }
            lam <- fit@lambda
            add(i = b - 1, j = a - 1, order = ord, variant = v, spec = sp, K = K, type = ty,
                stat = r0(fit), lam1 = lam[1], lam2 = lam[2])
          }
        }
      }
    }
  }
} else if (mode == "df") {
  for (d in 1:P) for (r in 1:P) {
    if (d == r) next
    y0 <- X[, d]; y1 <- X[, r]
    for (tr in c("c", "ct")) {
      u <- if (tr == "c") residuals(lm(y0 ~ y1)) else residuals(lm(y0 ~ y1 + tt))
      for (p in 0:10) {
        f <- ur.df(u, type = "none", lags = p, selectlags = "Fixed")
        add(dep = d - 1, reg = r - 1, trend = tr, sel = "Fixed", p = p, tau = f@teststat[1])
      }
      for (s in c("AIC", "BIC")) {
        f <- ur.df(u, type = "none", lags = 10, selectlags = s)
        pl <- length(coef(f@testreg)[, 1]) - 1
        add(dep = d - 1, reg = r - 1, trend = tr, sel = s, p = pl, tau = f@teststat[1])
      }
    }
  }
} else stop("unknown mode")
out <- do.call(rbind, rows)
num <- vapply(out, is.numeric, logical(1))
out[num] <- lapply(out[num], function(v) sprintf("%.17g", v))
write.csv(out, fout, row.names = FALSE, quote = FALSE)
