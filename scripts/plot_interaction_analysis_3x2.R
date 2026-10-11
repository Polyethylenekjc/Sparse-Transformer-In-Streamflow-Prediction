#!/usr/bin/env Rscript

required_pkgs <- c("readr", "dplyr", "tidyr", "ggplot2", "patchwork", "jsonlite", "maps")
missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  stop(
    sprintf(
      "Missing R packages: %s\nInstall with:\ninstall.packages(c(%s))",
      paste(missing_pkgs, collapse = ", "),
      paste(sprintf('"%s"', missing_pkgs), collapse = ", ")
    )
  )
}

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(patchwork)
  library(jsonlite)
  library(maps)
})

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  out <- list(
    output_root = "/mnt/d/Store/Sparse_Transformer",
    stations_csv = "data/stations.csv",
    out_figure = "paper/Figures/interaction_analysis.png",
    out_table = "outputs/figures/interaction_analysis_subset.csv",
    width = 14,
    height = 17,
    dpi = 320
  )

  if (length(args) == 0) {
    return(out)
  }

  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (key == "--output-root" && i + 1 <= length(args)) {
      out$output_root <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--stations-csv" && i + 1 <= length(args)) {
      out$stations_csv <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--out-figure" && i + 1 <= length(args)) {
      out$out_figure <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--out-table" && i + 1 <= length(args)) {
      out$out_table <- args[[i + 1]]
      i <- i + 2
    } else if (key == "--width" && i + 1 <= length(args)) {
      out$width <- as.numeric(args[[i + 1]])
      i <- i + 2
    } else if (key == "--height" && i + 1 <= length(args)) {
      out$height <- as.numeric(args[[i + 1]])
      i <- i + 2
    } else if (key == "--dpi" && i + 1 <= length(args)) {
      out$dpi <- as.integer(args[[i + 1]])
      i <- i + 2
    } else {
      stop(sprintf("Unknown or incomplete argument: %s", key))
    }
  }

  out
}

resolve_output_root <- function(cli_path) {
  candidates <- c(
    trimws(Sys.getenv("SPARSE_TRANSFORMER_OUTPUT_ROOT", unset = "")),
    trimws(cli_path),
    "/mnt/d/Store/Sparse_Transformer",
    "outputs"
  )
  for (p in candidates) {
    if (nzchar(p) && dir.exists(p)) {
      return(normalizePath(path.expand(p), winslash = "/", mustWork = FALSE))
    }
  }
  normalizePath(path.expand("outputs"), winslash = "/", mustWork = FALSE)
}

pad_station_id <- function(x) {
  num <- suppressWarnings(as.integer(as.character(x)))
  ifelse(is.na(num), NA_character_, sprintf("%08d", num))
}

safe_read_json <- function(path) {
  if (!file.exists(path)) return(NULL)
  tryCatch(jsonlite::fromJSON(path), error = function(e) NULL)
}

safe_read_csv <- function(path) {
  if (!file.exists(path)) return(NULL)
  tryCatch(readr::read_csv(path, show_col_types = FALSE), error = function(e) NULL)
}

discover_station_dirs <- function(output_root) {
  if (!dir.exists(output_root)) return(character())
  dirs <- list.dirs(output_root, recursive = FALSE, full.names = TRUE)
  dirs[file.exists(file.path(dirs, "summary.json"))]
}

abbr_feature <- function(name) {
  low <- tolower(trimws(as.character(name)))
  if (low == "target_q") return("Qtar")
  if (grepl("streamflow|^q$|flow|runoff", low)) return("Q")
  if (grepl("prcp|precip|rain", low)) return("P")
  if (grepl("tmax", low)) return("Tmax")
  if (grepl("tmin", low)) return("Tmin")
  if (grepl("srad", low)) return("SRAD")
  if (grepl("pet", low)) return("PET")
  if (grepl("temp|^t$", low)) return("T")
  if (grepl("dayl", low)) return("DayL")
  if (grepl("swe", low)) return("SWE")
  if (grepl("vp", low)) return("VP")
  if (grepl("soil", low)) return("Soil")
  parts <- unlist(strsplit(gsub("-", "_", low), "_"), use.names = FALSE)
  parts <- parts[nzchar(parts)]
  if (length(parts) == 0) return(substr(name, 1, 5))
  paste0(toupper(substr(parts, 1, 1)), collapse = "")
}

classify_interaction_pair <- function(var_i, var_j) {
  a <- abbr_feature(var_i)
  b <- abbr_feature(var_j)
  pair <- sort(unique(c(a, b)))

  if (identical(pair, c("Q", "Qtar"))) return(list(class_id = "qqtar", label = "Q x Qtar"))
  if (identical(pair, c("P", "Q"))) return(list(class_id = "pq", label = "P x Q"))
  if (length(intersect(pair, c("Q"))) == 1 && length(intersect(pair, c("Tmin", "Tmax", "SRAD"))) == 1) {
    return(list(class_id = "tq", label = "T/SRAD x Q"))
  }
  list(class_id = NA_character_, label = NA_character_)
}

tq_subtype_from_pair <- function(var_i, var_j) {
  a <- abbr_feature(var_i)
  b <- abbr_feature(var_j)
  pair <- c(a, b)
  if ("Tmin" %in% pair) return("Tmin")
  if ("Tmax" %in% pair) return("Tmax")
  if ("SRAD" %in% pair) return("SRAD")
  "Other"
}

extract_probe_r2 <- function(probe_obj, key) {
  if (is.null(probe_obj) || !is.list(probe_obj)) return(NA_real_)
  obj <- probe_obj[[key]]
  if (!is.list(obj) || is.null(obj$r2)) return(NA_real_)
  val <- suppressWarnings(as.numeric(obj$r2))
  if (!is.finite(val) || val < 0 || val > 1) return(NA_real_)
  val
}

size_from_nse <- function(nse, global_min, global_max) {
  if (!is.finite(nse)) return(2.4)
  if (!is.finite(global_min) || !is.finite(global_max) || global_max <= global_min) return(2.4)
  1.8 + 3.4 * (nse - global_min) / (global_max - global_min)
}

build_station_table <- function(output_root, stations_csv) {
  station_dirs <- discover_station_dirs(output_root)
  if (length(station_dirs) == 0) stop(sprintf("No station summary.json files found under: %s", output_root))

  rows <- list()
  for (sd in station_dirs) {
    sid <- pad_station_id(basename(sd))
    summary_obj <- safe_read_json(file.path(sd, "summary.json"))
    if (is.null(summary_obj) || is.null(summary_obj$metrics)) next

    ranking_df <- safe_read_csv(file.path(sd, "variable_interactions_ranking.csv"))
    if (is.null(ranking_df) || nrow(ranking_df) == 0) next

    ranking_df$interaction <- suppressWarnings(as.numeric(ranking_df$interaction))
    ranking_df <- ranking_df[is.finite(ranking_df$interaction), ]
    if (nrow(ranking_df) == 0) next

    top_row <- ranking_df[which.max(abs(ranking_df$interaction)), , drop = FALSE][1, ]
    cls <- classify_interaction_pair(top_row$var_i, top_row$var_j)
    if (is.na(cls$class_id)) next

    probe_obj <- safe_read_json(file.path(sd, "physical_probe_results.json"))
    metrics <- summary_obj$metrics

    rows[[length(rows) + 1]] <- data.frame(
      station_id = sid,
      nse = suppressWarnings(as.numeric(metrics$nse)),
      top_pair_interaction = suppressWarnings(as.numeric(top_row$interaction)),
      tq_subtype = tq_subtype_from_pair(top_row$var_i, top_row$var_j),
      probe_r2_cum_rain = extract_probe_r2(probe_obj, "cum_rain"),
      probe_r2_api = extract_probe_r2(probe_obj, "api"),
      probe_r2_dpdt = extract_probe_r2(probe_obj, "dpdt"),
      interaction_class = cls$class_id,
      stringsAsFactors = FALSE
    )
  }

  if (length(rows) == 0) stop(sprintf("No usable station interaction tables found under: %s", output_root))

  feat_df <- dplyr::bind_rows(rows)
  stations_df <- safe_read_csv(stations_csv)
  if (is.null(stations_df)) stop(sprintf("stations.csv not found or unreadable: %s", stations_csv))
  if (!all(c("gauge_id", "lat", "lon") %in% names(stations_df))) stop("stations.csv must contain gauge_id, lat, lon")

  stations_df <- stations_df %>%
    transmute(
      station_id = pad_station_id(gauge_id),
      lat = suppressWarnings(as.numeric(lat)),
      lon = suppressWarnings(as.numeric(lon))
    )

  out <- feat_df %>% left_join(stations_df, by = "station_id") %>% filter(is.finite(lat), is.finite(lon))
  if (nrow(out) == 0) stop("No stations matched to valid lat/lon")
  out
}

make_blank_panel <- function(title_text, tag_text, message_text, family_use) {
  ggplot() +
    annotate("text", x = 0.5, y = 0.5, label = message_text, family = family_use, size = 5.2) +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
    labs(title = title_text, tag = tag_text) +
    theme_void(base_family = family_use) +
    theme(
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      plot.title = element_text(face = "bold", size = 14.5, hjust = 0.5),
      plot.tag = element_text(face = "bold", size = 16.5)
    )
}

make_map_panel <- function(df, title_text, tag_text, color_limits, global_nse_limits, family_use, row_palette, legend_title, class_id) {
  if (nrow(df) == 0) return(make_blank_panel(title_text, tag_text, "No stations", family_use))

  df <- df %>%
    mutate(
      nse = as.numeric(nse),
      nse_vis = pmin(pmax(nse, color_limits[1]), color_limits[2]),
      point_size = vapply(nse, size_from_nse, numeric(1), global_min = global_nse_limits[1], global_max = global_nse_limits[2])
    )

  usa_map <- ggplot2::map_data("state")
  p <- ggplot() +
    geom_polygon(data = usa_map, aes(x = long, y = lat, group = group), fill = "#f5f4f3", color = "grey55", linewidth = 0.48) +
    scale_size_identity(guide = "none") +
    scale_fill_gradientn(
      colors = row_palette,
      limits = color_limits,
      oob = scales::squish,
      name = legend_title,
      guide = guide_colorbar(title.position = "top", direction = "horizontal")
    ) +
    coord_quickmap(xlim = c(-125, -66), ylim = c(24, 50), expand = FALSE) +
    labs(title = title_text, x = NULL, y = NULL, tag = tag_text) +
    theme_classic(base_family = family_use, base_size = 14.8) +
    theme(
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.title = element_blank(),
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      axis.line = element_blank(),
      panel.border = element_blank(),
      plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
      plot.tag = element_text(face = "bold", size = 18),
      legend.position = "bottom",
      legend.justification = c(0.5, 0.5),
      legend.background = element_rect(fill = "white", color = NA)
    )

  if (class_id == "tq") {
    df <- df %>% mutate(tq_subtype = factor(tq_subtype, levels = c("Tmin", "Tmax", "SRAD", "Other")))
    p <- p +
      geom_point(data = df, aes(x = lon, y = lat, fill = nse_vis, shape = tq_subtype, size = point_size), alpha = 0.95, color = "black", stroke = 0.28) +
      scale_shape_manual(values = c(Tmin = 21, Tmax = 24, SRAD = 22, Other = 23), breaks = c("Tmin", "Tmax", "SRAD"), name = "T/SRAD type") +
      guides(shape = guide_legend(order = 2, override.aes = list(fill = row_palette[1:3], color = "black")), fill = guide_colorbar(order = 1))
  } else {
    p <- p + geom_point(data = df, aes(x = lon, y = lat, fill = nse_vis, size = point_size), alpha = 0.95, shape = 21, color = "black", stroke = 0.28)
  }

  p
}

make_box_group_panel <- function(df, title_text, family_use, row_main_color, row_probe_colors) {
  if (nrow(df) == 0) return(make_blank_panel(title_text, "", "No stations", family_use))

  nse_df <- df %>%
    transmute(NSE = as.numeric(nse), `Interaction strength` = abs(as.numeric(top_pair_interaction))) %>%
    pivot_longer(cols = c("NSE", "Interaction strength"), names_to = "metric", values_to = "value") %>%
    mutate(metric = factor(metric, levels = c("NSE", "Interaction strength"))) %>%
    filter(is.finite(value))

  df_nse <- nse_df %>% filter(metric == "NSE") %>% mutate(metric_x = 1.00)
  df_int <- nse_df %>% filter(metric == "Interaction strength") %>% mutate(metric_x = 1.42)

  nse_lim <- range(df_nse$value, finite = TRUE)
  int_lim <- range(df_int$value, finite = TRUE)
  if (!all(is.finite(nse_lim)) || diff(nse_lim) <= 0) nse_lim <- c(0, 1)
  if (!all(is.finite(int_lim)) || diff(int_lim) <= 0) int_lim <- c(0, 1)

  map_int_to_nse <- function(x) {
    (x - int_lim[1]) / (int_lim[2] - int_lim[1]) * (nse_lim[2] - nse_lim[1]) + nse_lim[1]
  }
  map_nse_to_int <- function(y) {
    (y - nse_lim[1]) / (nse_lim[2] - nse_lim[1]) * (int_lim[2] - int_lim[1]) + int_lim[1]
  }

  p_nse <- ggplot() +
    geom_boxplot(data = df_nse, aes(x = metric_x, y = value), width = 0.28, outlier.size = 0.95, outlier.alpha = 0.34, linewidth = 0.38, color = "grey18", fill = row_main_color) +
    geom_boxplot(data = df_int, aes(x = metric_x, y = map_int_to_nse(value)), width = 0.28, outlier.size = 0.95, outlier.alpha = 0.34, linewidth = 0.38, color = "grey18", fill = scales::alpha(row_main_color, 0.7)) +
    scale_x_continuous(
      breaks = c(1.00, 1.42),
      labels = c("NSE", "Strength"),
      expand = expansion(add = c(0.18, 0.18))
    ) +
    scale_y_continuous(name = "NSE", sec.axis = sec_axis(~ map_nse_to_int(.), name = "Interaction strength")) +
    labs(title = "NSE and Interaction Strength", x = NULL, y = NULL) +
    theme_classic(base_family = family_use, base_size = 14.0) +
    theme(
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      panel.grid.major = element_line(color = "grey88", linewidth = 0.22),
      panel.grid.minor = element_blank(),
      axis.text.x = element_text(size = 11.2, angle = 0, hjust = 0.5),
      axis.text.y = element_text(size = 11.4),
      axis.title.y.right = element_text(size = 11.4),
      plot.title = element_text(face = "bold", size = 14.2, hjust = 0.5)
    )

  probe_df <- df %>%
    transmute(`CUM rain` = as.numeric(probe_r2_cum_rain), `API` = as.numeric(probe_r2_api), `dP/dt` = as.numeric(probe_r2_dpdt)) %>%
    pivot_longer(cols = c("CUM rain", "API", "dP/dt"), names_to = "metric", values_to = "value") %>%
    filter(is.finite(value)) %>%
    mutate(metric = factor(metric, levels = c("API", "CUM rain", "dP/dt")),
           metric_x = dplyr::case_when(
             metric == "API" ~ 1.00,
             metric == "CUM rain" ~ 1.42,
             TRUE ~ 1.84
           ))

  p_probe <- ggplot(probe_df, aes(x = metric_x, y = value, fill = metric)) +
    geom_boxplot(width = 0.28, outlier.size = 0.95, outlier.alpha = 0.34, linewidth = 0.38, color = "grey18") +
    scale_fill_manual(
      values = c("CUM rain" = row_probe_colors[1], "API" = row_probe_colors[2], "dP/dt" = row_probe_colors[3]),
      guide = "none"
    ) +
    scale_x_continuous(
      breaks = c(1.00, 1.42, 1.84),
      labels = c("API", "CUM rain", "dP/dt"),
      expand = expansion(add = c(0.18, 0.18))
    ) +
    coord_cartesian(ylim = c(0, 1)) +
    labs(title = "Physical Probe States", x = NULL, y = expression(R^2)) +
    theme_classic(base_family = family_use, base_size = 12.8) +
    theme(
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      panel.grid.major = element_line(color = "grey88", linewidth = 0.22),
      panel.grid.minor = element_blank(),
      axis.text.x = element_text(size = 11.2, angle = 0, hjust = 0.5),
      axis.text.y = element_text(size = 11.4),
      plot.title = element_text(face = "bold", size = 14.2, hjust = 0.5)
    )

  (p_nse | p_probe) +
    patchwork::plot_layout(widths = c(0.40, 0.60)) +
    patchwork::plot_annotation(title = title_text) &
    theme(
      plot.title = element_text(face = "bold", size = 12.4, hjust = 0.5),
      plot.tag = element_text(face = "bold", size = 15)
    )
}

main <- function() {
  args <- parse_args()
  output_root <- resolve_output_root(args$output_root)
  stations_csv <- normalizePath(path.expand(args$stations_csv), winslash = "/", mustWork = FALSE)
  if (!file.exists(stations_csv)) stop(sprintf("stations.csv not found: %s", stations_csv))

  family_use <- "Times New Roman"
  if (requireNamespace("systemfonts", quietly = TRUE)) {
    fams <- tryCatch(systemfonts::system_fonts()$family, error = function(e) character())
    if (!any(tolower(fams) == tolower("Times New Roman"))) family_use <- "serif"
  }

  df <- build_station_table(output_root = output_root, stations_csv = stations_csv)
  nse_threshold <- 0.5
  df_all <- df %>%
    filter(interaction_class %in% c("qqtar", "pq", "tq")) %>%
    filter(is.finite(as.numeric(nse)), as.numeric(nse) > nse_threshold)
  if (nrow(df_all) == 0) stop(sprintf("No stations matched target classes with NSE > %.2f", nse_threshold))

  global_nse_limits <- range(as.numeric(df_all$nse), finite = TRUE)
  if (!is.finite(global_nse_limits[1]) || !is.finite(global_nse_limits[2]) || global_nse_limits[1] == global_nse_limits[2]) {
    global_nse_limits <- c(0, 1)
  }

  global_color_limits <- range(as.numeric(df_all$top_pair_interaction), finite = TRUE)
  if (!is.finite(global_color_limits[1]) || !is.finite(global_color_limits[2]) || global_color_limits[1] == global_color_limits[2]) {
    global_color_limits <- c(-1, 1)
  }

  row_defs <- list(
    list(class_id = "qqtar", map_title = "(a) Q x Qtar interaction", box_title = "(b) Statistics by interaction group", row_main = "#2b6cb0", row_probe = c("#2b6cb0", "#4a84c2", "#6a9bd4"), map_pal = c("#fcf8f2", "#9fb8de", "#203c7c"), leg = "NSE"),
    list(class_id = "pq", map_title = "(c) P x Q interaction", box_title = "(d) Statistics by interaction group", row_main = "#f97316", row_probe = c("#f97316", "#fb8c3b", "#fda76a"), map_pal = c("#fff5eb", "#fdae6b", "#b33d00"), leg = "NSE"),
    list(class_id = "tq", map_title = "(e) Tmin/Tmax/SRAD x Q interaction", box_title = "(f) Statistics by interaction group", row_main = "#8e63b6", row_probe = c("#6a3d9a", "#8e63b6", "#b07cc6"), map_pal = c("#f7f2fb", "#b79ad7", "#5a2a86"), leg = "NSE")
  )

  panel_plots <- list()
  for (rd in row_defs) {
    df_row <- df_all %>% filter(interaction_class == rd$class_id)
    p_map <- make_map_panel(df_row, rd$map_title, "", global_nse_limits, global_nse_limits, family_use, rd$map_pal, rd$leg, rd$class_id)
    p_box <- make_box_group_panel(df_row, rd$box_title, family_use, rd$row_main, rd$row_probe)
    panel_plots[[length(panel_plots) + 1]] <- p_map
    panel_plots[[length(panel_plots) + 1]] <- p_box
  }

  fig <- patchwork::wrap_plots(panel_plots, ncol = 2) + patchwork::plot_layout(guides = "keep", widths = c(1.65, 1.00)) &
    theme(plot.background = element_rect(fill = "white", color = NA))

  out_figure <- normalizePath(path.expand(args$out_figure), winslash = "/", mustWork = FALSE)
  out_table <- normalizePath(path.expand(args$out_table), winslash = "/", mustWork = FALSE)
  dir.create(dirname(out_figure), recursive = TRUE, showWarnings = FALSE)
  dir.create(dirname(out_table), recursive = TRUE, showWarnings = FALSE)

  ggplot2::ggsave(filename = out_figure, plot = fig, width = args$width, height = args$height, dpi = args$dpi, limitsize = FALSE, bg = "white")
  readr::write_csv(df_all, out_table)

  message(sprintf("Output root: %s", output_root))
  message(sprintf("NSE filter: > %.2f", nse_threshold))
  message(sprintf("Stations matched: %d", nrow(df_all)))
  message(sprintf("Figure saved: %s", out_figure))
  message(sprintf("Subset table saved: %s", out_table))
}

main()
