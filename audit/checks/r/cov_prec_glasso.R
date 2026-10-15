# External reference for the cov_prec_glasso family: R package glasso (Friedman, Hastie & Tibshirani).
#
# Invoked by refs/cov_prec_glasso.py through subprocess with R_LIBS_USER=/home/dan/R/audit-lib:
#   Rscript r/cov_prec_glasso.R <exchange_dir>
# Reads <exchange_dir>/jobs.csv (columns: name, alpha, penalize_diagonal, thr, maxit) and, per job,
# <exchange_dir>/<name>_S.csv (the P x P empirical covariance, no header, written with 17 significant
# digits by the Python side). Writes <name>_w.csv (estimated covariance W), <name>_wi.csv (estimated
# inverse covariance Theta) and <exchange_dir>/r_summary.csv (name, niter, glasso version, R version).
#
# Convention notes (documented before any comparison):
# * glasso() implements Friedman, Hastie & Tibshirani 2008 (Biostatistics 9:432, eq. 1), which maximises
#   log det(Theta) - tr(S Theta) - rho * ||Theta||_1 (the ?glasso help cites this paper), i.e. the
#   same objective as sklearn's _graphical_lasso (minimise -log det + tr(S Theta) + alpha * sum|Theta_ij|,
#   sklearn/covariance/_graph_lasso.py:43-53) with rho = alpha: no factor 2, no factor T.
# * penalize.diagonal = FALSE excludes the diagonal from the penalty, which is the sklearn convention
#   (diagonal unpenalised, W_ii = S_ii). penalize.diagonal = TRUE (the glasso default and Friedman 2008
#   eq. 1) gives W_ii = S_ii + rho; jobs with penalize_diagonal = 1 document that difference.
# * S is passed directly (glasso does not centre or rescale it), so both packages solve the same problem.
suppressPackageStartupMessages(library(glasso))

args <- commandArgs(trailingOnly = TRUE)
ex <- args[1]
jobs <- read.csv(file.path(ex, "jobs.csv"), stringsAsFactors = FALSE)
out <- data.frame(name = character(0), niter = numeric(0), stringsAsFactors = FALSE)
for (k in seq_len(nrow(jobs))) {
  j <- jobs[k, ]
  S <- as.matrix(read.csv(file.path(ex, paste0(j$name, "_S.csv")), header = FALSE))
  dimnames(S) <- NULL
  fit <- glasso(S, rho = j$alpha, penalize.diagonal = as.logical(j$penalize_diagonal),
                thr = j$thr, maxit = j$maxit)
  write.table(format(fit$w, digits = 17), file.path(ex, paste0(j$name, "_w.csv")),
              sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)
  write.table(format(fit$wi, digits = 17), file.path(ex, paste0(j$name, "_wi.csv")),
              sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)
  out <- rbind(out, data.frame(name = j$name, niter = fit$niter, stringsAsFactors = FALSE))
}
out$glasso_version <- as.character(packageVersion("glasso"))
out$r_version <- R.version.string
write.csv(out, file.path(ex, "r_summary.csv"), row.names = FALSE)
