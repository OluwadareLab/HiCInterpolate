#!/usr/bin/env Rscript
# Run FLAMINGOrLite::flamingo_basic on a square IF matrix and write coords.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: flamingo_basic_run.R <matrix.txt> <output_dir> [max_iter]")
}

matrix_file <- args[[1]]
output_dir <- args[[2]]
max_iter <- if (length(args) >= 3) as.integer(args[[3]]) else 500L

if (!requireNamespace("FLAMINGOrLite", quietly = TRUE)) {
  stop("FLAMINGOrLite not installed. Install with:\n",
       "  remotes::install_github('JiaxinYangJX/FLAMINGOrLite')")
}

suppressPackageStartupMessages(library(FLAMINGOrLite))

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

input_if <- as.matrix(read.table(matrix_file, header = FALSE))
n <- nrow(input_if)
if (n != ncol(input_if)) {
  stop(sprintf("Expected square matrix, got %d x %d", n, ncol(input_if)))
}
if (n > 200) {
  warning(sprintf(
    "Matrix has %d bins (>200). flamingo_basic is intended for <=200 fragments; consider flamingo_main.",
    n
  ))
}

pred <- flamingo_basic(
  input_if = input_if,
  sample_rate = 0.75,
  lambda = 10,
  r = 1,
  max_dist = 0.1,
  error_threshold = 1e-4,
  max_iter = max_iter,
  alpha = -0.25,
  inf_dist = 4
)

if (is.null(pred)) {
  stop("flamingo_basic returned NULL (no consecutive contacts on sub-diagonal)")
}

coords <- pred@coordinates
ids <- pred@id
# coords may be all points or only valid; align with id
if (nrow(coords) == length(ids)) {
  xyz <- coords
} else if (nrow(coords) == pred@input_n) {
  xyz <- coords[ids, , drop = FALSE]
} else {
  stop(sprintf(
    "Unexpected coordinate shape %s vs id length %d / input_n %d",
    paste(dim(coords), collapse = "x"), length(ids), pred@input_n
  ))
}

out_tsv <- file.path(output_dir, "flamingo_coords.tsv")
df <- data.frame(
  id = ids,
  x = xyz[, 1],
  y = xyz[, 2],
  z = xyz[, 3]
)
write.table(df, out_tsv, sep = "\t", row.names = FALSE, quote = FALSE)

vtk_path <- file.path(output_dir, "flamingo_structure.vtk")
# write.vtk uses dim(points)[1] as N — pass N x 3
write.vtk(
  points = xyz,
  lookup_table = ids,
  name = "flamingo",
  opt_path = vtk_path
)

message("Wrote ", out_tsv)
message("Wrote ", vtk_path)
