#!/usr/bin/env Rscript
# FLAMINGOr runner (wangjr03/FLAMINGO).
#
# Modes:
#   basic <if_matrix.txt> <output_dir> [sample_rate] [lambda] [max_dist] [nThread] [alpha]
#   dense_large <if_matrix.txt> <output_dir> <domain_res> <frag_res> <chr_name>
#               <downsampling_rates> <lambda> <max_dist> <nThread> [max_iter] [alpha]
#   large <file_format> <hic_data_low> <output_dir> <domain_res> <frag_res> <chr_size>
#         <chr_name> <normalization> <downsampling_rates> <lambda> <max_dist> <nThread>
#         [hic_data_high] [norm_low] [norm_high] [max_iter] [alpha]

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: flamingo_main_run.R <basic|dense_large|large> ...")
}

mode <- args[[1]]

if (!requireNamespace("FLAMINGOr", quietly = TRUE)) {
  stop("FLAMINGOr not installed. Install with:\n",
       "  remotes::install_local('/path/to/FLAMINGO/FLAMINGOr')")
}

suppressPackageStartupMessages({
  library(FLAMINGOr)
  library(Matrix)
})

aggregate_if <- function(mat, factor) {
  if (factor <= 1) return(mat)
  n <- (nrow(mat) %/% factor) * factor
  if (n == 0) stop("Matrix too small for domain factor ", factor)
  cropped <- mat[seq_len(n), seq_len(n), drop = FALSE]
  m <- n %/% factor
  out <- matrix(0, m, m)
  for (i in seq_len(m)) {
    ir <- ((i - 1L) * factor + 1L):(i * factor)
    for (j in i:m) {
      jr <- ((j - 1L) * factor + 1L):(j * factor)
      s <- sum(cropped[ir, jr])
      out[i, j] <- s
      out[j, i] <- s
    }
  }
  out
}

make_flamingo_obj <- function(input_if, chr_name, alpha) {
  input_if <- as.matrix(input_if)
  input_if[!is.finite(input_if)] <- 0
  pd <- input_if^alpha
  new("flamingo", IF = input_if, PD = pd, n_frag = nrow(input_if), chr_name = chr_name)
}

run_hierarchical <- function(flamingo_low, flamingo_high, domain_res, frag_res,
                             downsampling_rates, lambda, max_dist, nThread, max_iter) {
  print("Dividing domains...")
  flamingo.divide_domain(flamingo_obj = flamingo_high, domain_res = domain_res, frag_res = frag_res)

  print("Reconstructing backbones...")
  flamingo_backbone_prediction <- flamingo.reconstruct_backbone_structure(
    flamingo_data_obj = flamingo_low,
    sw = downsampling_rates,
    lambda = lambda,
    max_dist = max_dist,
    nThread = 1
  )
  print("Reconstructing intra-domain structures...")
  flamigo_intra_domain_prediction <- flamingo.reconstruct_structure(
    sw = downsampling_rates,
    lambda = lambda,
    max_dist = max_dist,
    nThread = nThread
  )
  save(flamigo_intra_domain_prediction, file = "intra_domain.Rdata")

  print("Assembling structures...")
  flamingo.assemble_structure(
    flamingo_backbone_prediction_obj = flamingo_backbone_prediction,
    flamingo_final_res_data_obj = flamingo_high,
    list_of_flamingo_domain_prediction_obj = flamigo_intra_domain_prediction,
    max_iter = max_iter
  )
}

write_outputs_df <- function(res, output_dir, label) {
  out_tsv <- file.path(output_dir, "flamingo_coords.tsv")
  write.table(res, out_tsv, sep = "\t", row.names = FALSE, quote = FALSE)
  message("Wrote ", out_tsv)

  cn <- tolower(colnames(res))
  if (all(c("x", "y", "z") %in% cn)) {
    xyz <- as.matrix(res[, match(c("x", "y", "z"), cn), drop = FALSE])
    ids <- if ("frag_id" %in% cn) res[[match("frag_id", cn)]] else seq_len(nrow(xyz))
  } else if (ncol(res) >= 4) {
    xyz <- as.matrix(res[, (ncol(res) - 2):ncol(res), drop = FALSE])
    ids <- res[, 1]
  } else {
    stop("Cannot find x/y/z columns in FLAMINGO output")
  }
  storage.mode(xyz) <- "double"
  ok <- rowSums(is.finite(xyz)) == 3L
  xyz <- xyz[ok, , drop = FALSE]
  ids <- ids[ok]
  write.vtk(
    points = xyz,
    lookup_table = ids,
    name = label,
    opt_path = file.path(output_dir, "flamingo_structure.vtk")
  )
  message("Wrote ", file.path(output_dir, "flamingo_structure.vtk"))
}

write_outputs_pred <- function(xyz, ids, output_dir, label) {
  out_tsv <- file.path(output_dir, "flamingo_coords.tsv")
  df <- data.frame(frag_id = ids, x = xyz[, 1], y = xyz[, 2], z = xyz[, 3])
  write.table(df, out_tsv, sep = "\t", row.names = FALSE, quote = FALSE)
  message("Wrote ", out_tsv)
  write.vtk(
    points = xyz,
    lookup_table = ids,
    name = label,
    opt_path = file.path(output_dir, "flamingo_structure.vtk")
  )
  message("Wrote ", file.path(output_dir, "flamingo_structure.vtk"))
}

if (mode == "basic") {
  if (length(args) < 3) {
    stop("Usage: flamingo_main_run.R basic <if_matrix.txt> <output_dir> [sample_rate] [lambda] [max_dist] [nThread] [alpha]")
  }
  matrix_file <- args[[2]]
  output_dir <- args[[3]]
  sample_rate <- if (length(args) >= 4) as.numeric(args[[4]]) else 0.75
  lambda <- if (length(args) >= 5) as.numeric(args[[5]]) else 10
  max_dist <- if (length(args) >= 6) as.numeric(args[[6]]) else 0.01
  nThread <- if (length(args) >= 7) as.integer(args[[7]]) else 1L
  alpha <- if (length(args) >= 8) as.numeric(args[[8]]) else -0.25

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  input_if <- as.matrix(read.table(matrix_file, header = FALSE))
  n <- nrow(input_if)
  if (n != ncol(input_if)) stop(sprintf("Expected square matrix, got %d x %d", n, ncol(input_if)))
  if (n > 200) {
    warning(sprintf("Matrix has %d bins; basic mode is intended for <=200.", n))
  }

  input_pd <- input_if^alpha
  pred <- flamingo.reconstruct_structure_worker(
    input_if, input_pd, sample_rate, lambda, max_dist, nThread
  )
  if (is.null(pred)) stop("flamingo.reconstruct_structure_worker returned NULL")

  coords <- as.matrix(pred@coordinates)
  ids <- pred@id
  if (nrow(coords) == length(ids)) {
    xyz <- coords
  } else if (nrow(coords) == pred@input_n) {
    xyz <- coords[ids, , drop = FALSE]
  } else {
    stop(sprintf("Unexpected coordinate shape %s", paste(dim(coords), collapse = "x")))
  }
  storage.mode(xyz) <- "double"
  write_outputs_pred(xyz, ids, output_dir, "flamingo_basic")

} else if (mode == "dense_large") {
  if (length(args) < 10) {
    stop(paste(
      "Usage: flamingo_main_run.R dense_large <if_matrix.txt> <output_dir>",
      "<domain_res> <frag_res> <chr_name>",
      "<downsampling_rates> <lambda> <max_dist> <nThread> [max_iter] [alpha]"
    ))
  }
  matrix_file <- args[[2]]
  output_dir <- args[[3]]
  domain_res <- as.numeric(args[[4]])
  frag_res <- as.numeric(args[[5]])
  chr_name <- args[[6]]
  downsampling_rates <- as.numeric(args[[7]])
  lambda <- as.numeric(args[[8]])
  max_dist <- as.numeric(args[[9]])
  nThread <- as.integer(args[[10]])
  max_iter <- if (length(args) >= 11) as.integer(args[[11]]) else 500L
  alpha <- if (length(args) >= 12) as.numeric(args[[12]]) else -0.25

  if (domain_res %% frag_res != 0) {
    stop("domain_res must be an integer multiple of frag_res")
  }
  factor <- as.integer(domain_res / frag_res)

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  old_wd <- getwd()
  setwd(output_dir)
  on.exit(setwd(old_wd), add = TRUE)

  matrix_file <- normalizePath(
    if (startsWith(matrix_file, "/")) matrix_file else file.path(old_wd, matrix_file),
    mustWork = TRUE
  )

  message("Reading dense IF matrix: ", matrix_file)
  if (grepl("\\.bin$", matrix_file, ignore.case = TRUE)) {
    con <- file(matrix_file, "rb")
    dims <- readBin(con, what = "integer", n = 2L, size = 4L, endian = "little")
    vals <- readBin(con, what = "double", n = as.integer(dims[1]) * as.integer(dims[2]),
                    size = 8L, endian = "little")
    close(con)
    input_if <- matrix(vals, nrow = dims[1], ncol = dims[2], byrow = TRUE)
  } else {
    input_if <- as.matrix(data.table::fread(matrix_file, header = FALSE))
  }
  n <- nrow(input_if)
  if (n != ncol(input_if)) stop(sprintf("Expected square matrix, got %d x %d", n, ncol(input_if)))
  message("FLAMINGOr dense_large n=", n, " domain_res=", domain_res,
          " frag_res=", frag_res, " factor=", factor)

  flamingo_high <- make_flamingo_obj(input_if, chr_name, alpha)
  flamingo_low <- make_flamingo_obj(aggregate_if(input_if, factor), chr_name, alpha)
  rm(input_if)
  gc()

  res <- run_hierarchical(
    flamingo_low, flamingo_high, domain_res, frag_res,
    downsampling_rates, lambda, max_dist, nThread, max_iter
  )
  write_outputs_df(res, output_dir, paste(chr_name, "FLAMINGO"))

} else if (mode == "large") {
  if (length(args) < 13) {
    stop(paste(
      "Usage: flamingo_main_run.R large",
      "<file_format> <hic_data_low> <output_dir>",
      "<domain_res> <frag_res> <chr_size> <chr_name>",
      "<normalization> <downsampling_rates> <lambda> <max_dist> <nThread>",
      "[hic_data_high] [norm_low] [norm_high] [max_iter] [alpha]"
    ))
  }
  file_format <- args[[2]]
  hic_data_low <- args[[3]]
  output_dir <- args[[4]]
  domain_res <- as.numeric(args[[5]])
  frag_res <- as.numeric(args[[6]])
  chr_size <- as.numeric(args[[7]])
  chr_name <- args[[8]]
  normalization <- args[[9]]
  downsampling_rates <- as.numeric(args[[10]])
  lambda <- as.numeric(args[[11]])
  max_dist <- as.numeric(args[[12]])
  nThread <- as.integer(args[[13]])
  hic_data_high <- if (length(args) >= 14 && nzchar(args[[14]]) && args[[14]] != "NA") args[[14]] else NULL
  norm_low <- if (length(args) >= 15 && nzchar(args[[15]]) && args[[15]] != "NA") args[[15]] else NULL
  norm_high <- if (length(args) >= 16 && nzchar(args[[16]]) && args[[16]] != "NA") args[[16]] else NULL
  max_iter <- if (length(args) >= 17) as.integer(args[[17]]) else 500L
  alpha <- if (length(args) >= 18) as.numeric(args[[18]]) else -0.25

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  old_wd <- getwd()
  setwd(output_dir)
  on.exit(setwd(old_wd), add = TRUE)

  abs_or_null <- function(p) {
    if (is.null(p)) return(NULL)
    if (startsWith(p, "/")) p else file.path(old_wd, p)
  }
  hic_data_low <- normalizePath(abs_or_null(hic_data_low), mustWork = TRUE)
  if (!is.null(hic_data_high)) hic_data_high <- normalizePath(abs_or_null(hic_data_high), mustWork = TRUE)
  if (!is.null(norm_low)) norm_low <- normalizePath(abs_or_null(norm_low), mustWork = TRUE)
  if (!is.null(norm_high)) norm_high <- normalizePath(abs_or_null(norm_high), mustWork = TRUE)

  message("FLAMINGOr flamingo.main_func_large format=", file_format)

  res <- flamingo.main_func_large(
    hic_data_low = hic_data_low,
    file_format = file_format,
    domain_res = domain_res,
    frag_res = frag_res,
    chr_size = chr_size,
    chr_name = chr_name,
    normalization = normalization,
    downsampling_rates = downsampling_rates,
    lambda = lambda,
    max_dist = max_dist,
    nThread = nThread,
    alpha = alpha,
    max_iter = max_iter,
    hic_data_high = hic_data_high,
    norm_low = norm_low,
    norm_high = norm_high
  )
  write_outputs_df(res, output_dir, paste(chr_name, "FLAMINGO"))

} else {
  stop("Unknown mode: ", mode, " (use basic, dense_large, or large)")
}
