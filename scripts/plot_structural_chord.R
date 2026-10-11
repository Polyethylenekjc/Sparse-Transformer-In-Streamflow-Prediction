#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(circlize)
})

parse_args <- function(args) {
  out <- list()
  i <- 1
  while (i <= length(args)) {
    k <- args[[i]]
    if (startsWith(k, "--") && i < length(args)) {
      key <- substring(k, 3)
      out[[key]] <- args[[i + 1]]
      i <- i + 2
    } else {
      i <- i + 1
    }
  }
  out
}

args <- parse_args(commandArgs(trailingOnly = TRUE))

pairs_path <- args[["pairs"]]
single_path <- args[["single"]]
out_path <- args[["out"]]
top_n <- if (!is.null(args[["top"]])) as.integer(args[["top"]]) else 20L

if (is.null(pairs_path) || is.null(out_path)) {
  stop("Usage: Rscript plot_structural_chord.R --pairs pairs.csv --single single.csv --out out.png [--title txt] [--top 20]")
}

if (!file.exists(pairs_path)) {
  stop(paste("pairs file not found:", pairs_path))
}

pairs <- read.csv(pairs_path, stringsAsFactors = FALSE)
if (!all(c("from", "to", "value") %in% colnames(pairs))) {
  stop("pairs.csv must contain columns: from,to,value")
}

pairs$value <- suppressWarnings(as.numeric(pairs$value))
pairs <- pairs[is.finite(pairs$value) & pairs$value > 0, ]
if (nrow(pairs) == 0) {
  stop("pairs.csv has no positive edge value")
}

pairs <- pairs[order(pairs$value, decreasing = TRUE), ]
if (nrow(pairs) > top_n) {
  pairs <- pairs[seq_len(top_n), ]
}

single <- NULL
if (!is.null(single_path) && file.exists(single_path)) {
  single <- read.csv(single_path, stringsAsFactors = FALSE)
  if (all(c("var", "single_strength") %in% colnames(single))) {
    single$single_strength <- suppressWarnings(as.numeric(single$single_strength))
    single <- single[is.finite(single$single_strength) & single$single_strength >= 0, ]
  } else {
    single <- NULL
  }
}

nodes <- sort(unique(c(as.character(pairs$from), as.character(pairs$to))))

# base matrix for chordDiagram
mat <- matrix(0, nrow = length(nodes), ncol = length(nodes), dimnames = list(nodes, nodes))
for (i in seq_len(nrow(pairs))) {
  a <- as.character(pairs$from[i])
  b <- as.character(pairs$to[i])
  v <- as.numeric(pairs$value[i])
  if (!is.na(v) && v > 0 && a %in% nodes && b %in% nodes) {
    mat[a, b] <- mat[a, b] + v
    mat[b, a] <- mat[b, a] + v
  }
}

# Fixed palette provided by user; recycle when node count > palette length.
user_palette <- c(
  "#7DC69B", "#9BD7F3", "#D5EAD9", "#D8EEFB",
  "#DCD7EB", "#F2A1A7", "#FBDDDD", "#FCE6CF"
)
node_cols <- rep(user_palette, length.out = length(nodes))
grid_col <- setNames(node_cols, nodes)

# link colors by normalized pair value
vmin <- min(pairs$value)
vmax <- max(pairs$value)
if (vmax > vmin) {
  p01 <- (pairs$value - vmin) / (vmax - vmin)
} else {
  p01 <- rep(0.8, nrow(pairs))
}

add_alpha <- function(cols, alpha_vals) {
  m <- col2rgb(cols)
  out <- character(length(cols))
  for (i in seq_along(cols)) {
    out[i] <- rgb(m[1, i] / 255, m[2, i] / 255, m[3, i] / 255, alpha = alpha_vals[i])
  }
  out
}

base_link_col <- unname(grid_col[as.character(pairs$from)])
base_link_col[is.na(base_link_col)] <- "#6D78B7"
link_alpha <- 0.45 + 0.45 * p01
link_col <- add_alpha(base_link_col, link_alpha)

# output device
ext <- tolower(tools::file_ext(out_path))
dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
if (ext == "pdf") {
  pdf(out_path, width = 8, height = 8)
} else if (ext == "svg") {
  svg(out_path, width = 8, height = 8)
} else {
  png(out_path, width = 2200, height = 2200, res = 300)
}

circos.clear()
par(family = "Times New Roman")
circos.par(start.degree = 90, gap.degree = 4, track.margin = c(0.01, 0.01), canvas.xlim = c(-1.2, 1.2), canvas.ylim = c(-1.2, 1.2))

chordDiagram(
  x = pairs[, c("from", "to", "value")],
  grid.col = grid_col,
  col = link_col,
  transparency = 0.0,
  directional = 0,
  annotationTrack = c("grid"),
  preAllocateTracks = list(track.height = 0.08)
)

circos.trackPlotRegion(
  track.index = 1,
  panel.fun = function(x, y) {
    sector.name <- get.cell.meta.data("sector.index")
    xlim <- get.cell.meta.data("xlim")
    ylim <- get.cell.meta.data("ylim")
    circos.text(
      mean(xlim),
      ylim[1] + 0.1,
      sector.name,
      facing = "clockwise",
      niceFacing = TRUE,
      adj = c(0, 0.5),
      cex = 1.05,
      family = "Times New Roman"
    )
  },
  bg.border = NA
)

circos.clear()
dev.off()

cat(paste0("Chord figure saved: ", out_path, "\n"))
