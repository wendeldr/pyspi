# rank_corr external cross-check: R stats::cor(method = "spearman" | "kendall").
# Invoked by refs/rank_corr.py: Rscript r/rank_corr.R <exchange_dir>
# <exchange_dir>/jobs.csv has columns job,method; in_<job>.csv is a T x P matrix (no header,
# values written with 17 significant digits); out_<job>.csv receives the P x P matrix
# cor(X, method = method) written with 17 significant digits.
# stats::cor "kendall" is tau-b (C_cor, src/library/stats/src/cov.c: sum of sign products over
# sqrt of the products of the non-tied sign counts); "spearman" ranks each column with
# rank(ties.method = "average") and then takes the Pearson correlation.
args <- commandArgs(trailingOnly = TRUE)
ex <- args[1]
jobs <- read.csv(file.path(ex, "jobs.csv"), stringsAsFactors = FALSE)
for (k in seq_len(nrow(jobs))) {
  X <- as.matrix(read.csv(file.path(ex, paste0("in_", jobs$job[k], ".csv")), header = FALSE))
  storage.mode(X) <- "double"
  C <- stats::cor(X, method = jobs$method[k])
  out <- matrix(sprintf("%.17g", C), nrow = nrow(C))
  write.table(out, file.path(ex, paste0("out_", jobs$job[k], ".csv")), sep = ",",
              row.names = FALSE, col.names = FALSE, quote = FALSE)
}
cat(sprintf("rank_corr.R: %d jobs, R %s\n", nrow(jobs), paste(R.version$major, R.version$minor, sep = ".")))
