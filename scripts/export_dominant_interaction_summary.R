# 加载必要的库
library(tidyverse)

# 1. 读取当前汇总表
data <- read.csv("dominant_interaction_summary.csv", check.names = FALSE)

required_cols <- c("sector", "interaction_group", "station_type", "count", "percent_within_group")
missing_cols <- setdiff(required_cols, names(data))
if (length(missing_cols) > 0) {
  stop(sprintf("Missing required columns: %s", paste(missing_cols, collapse = ", ")))
}

df <- data %>%
  transmute(
    sector = as.character(sector),
    station_type = as.character(station_type),
    interaction_group = as.character(interaction_group),
    count = as.numeric(count),
    percent_within_group = as.numeric(percent_within_group)
  ) %>%
  filter(is.finite(count), count > 0)

target_interactions <- c("Q*Qtar", "P*Q", "Tmin*Q", "Tmax*Q", "SRAD*Q")
df <- df %>%
  filter(interaction_group %in% target_interactions)

if ("snow_regime" %in% unique(df$sector) && "precipitation_class" %in% unique(df$sector)) {
  df <- df %>% filter(sector != "precipitation_class")
}

if (nrow(df) == 0) {
  stop("No rows found for target interaction groups with count > 0")
}

sector_levels <- unique(df$sector)

station_order_map <- list(
  aridity_class = c("Humid", "Sub-humid", "Arid"),
  precipitation_class = c("Low Precip", "Medium Precip", "High Precip"),
  climate_type = c("Oceanic", "Humid Continental", "Arid Continental", "Plateau/Mountain"),
  snow_regime = c("Rain-dominant", "Mixed", "Snow-influenced")
)

df <- df %>%
  group_by(sector) %>%
  mutate(
    station_rank = {
      order_vec <- station_order_map[[first(sector)]]
      if (is.null(order_vec)) {
        as.numeric(factor(station_type, levels = unique(station_type)))
      } else {
        known_rank <- match(station_type, order_vec)
        fallback_rank <- length(order_vec) + as.numeric(factor(station_type, levels = sort(unique(station_type))))
        ifelse(is.na(known_rank), fallback_rank, known_rank)
      }
    }
  ) %>%
  ungroup()

station_levels <- df %>%
  distinct(sector, station_type, station_rank) %>%
  arrange(factor(sector, levels = sector_levels), station_rank, station_type)

interaction_levels <- target_interactions[target_interactions %in% unique(df$interaction_group)]
base_palette <- c(
  "Q*Qtar" = "#8FB4DC",
  "P*Q" = "#7AC3DF",
  "Tmin*Q" = "#70CDBE",
  "Tmax*Q" = "#AC99D2",
  "SRAD*Q" = "#FFDD8E"
)
group_palette <- base_palette[interaction_levels]

bar_width <- 1.02
bar_step <- 1.15
station_gap <- 1.25
sector_gap <- 2.2

plot_rows <- list()
station_labels <- list()
station_frame_rows <- list()
sector_labels <- list()
sector_frame_rows <- list()
current_x <- 0

for (sec in sector_levels) {
  sec_df <- df %>% filter(sector == sec)
  sec_start <- NA_real_
  sec_end <- NA_real_

  sec_station_levels <- station_levels %>% filter(sector == sec) %>% pull(station_type)
  for (st in sec_station_levels) {
    st_df <- sec_df %>% filter(station_type == st) %>% arrange(factor(interaction_group, levels = interaction_levels))
    if (nrow(st_df) == 0) next

    st_start <- current_x + bar_step
    st_df <- st_df %>% mutate(x = seq(from = st_start, by = bar_step, length.out = nrow(st_df)))
    st_end <- max(st_df$x)

    if (is.na(sec_start)) sec_start <- st_start
    sec_end <- st_end

    plot_rows[[length(plot_rows) + 1]] <- st_df
    station_labels[[length(station_labels) + 1]] <- tibble(
      sector = sec,
      station_type = st,
      center = mean(c(st_start, st_end))
    )
    station_frame_rows[[length(station_frame_rows) + 1]] <- tibble(
      sector = sec,
      station_type = st,
      xmin = st_start - bar_width / 2,
      xmax = st_end + bar_width / 2
    )

    current_x <- st_end + station_gap
  }

  if (is.finite(sec_start) && is.finite(sec_end)) {
    sector_labels[[length(sector_labels) + 1]] <- tibble(
      sector = sec,
      center = mean(c(sec_start, sec_end))
    )
    sector_frame_rows[[length(sector_frame_rows) + 1]] <- tibble(
      sector = sec,
      xmin = sec_start - bar_width / 2,
      xmax = sec_end + bar_width / 2
    )
    current_x <- current_x + sector_gap
  }
}

df_plot <- bind_rows(plot_rows)
station_label_df <- bind_rows(station_labels)
station_frame_df <- bind_rows(station_frame_rows)
sector_label_df <- bind_rows(sector_labels)
sector_frame_df <- bind_rows(sector_frame_rows)

sector_bg_palette <- c("#F4EBC6", "#ECEFF1", "#E6F2EE")
sector_bg_df <- sector_frame_df %>%
  mutate(bg_fill = sector_bg_palette[((seq_len(n()) - 1) %% length(sector_bg_palette)) + 1])

station_type_levels <- unique(station_frame_df$station_type)
station_bg_map <- setNames(
  grDevices::hcl.colors(length(station_type_levels), palette = "Pastel 1"),
  station_type_levels
)
station_bg_df <- station_frame_df %>%
  mutate(bg_fill = station_bg_map[station_type])

max_value <- max(df_plot$count, na.rm = TRUE)
y_min <- min(df_plot$count[df_plot$count > 0], na.rm = TRUE)
ref_breaks <- unique(10^pretty(log10(c(y_min, max_value)), n = 5))
ref_breaks <- ref_breaks[ref_breaks >= y_min & ref_breaks <= max_value * 1.08]
ref_line_df <- tibble(y = ref_breaks)
ref_segment_df <- tidyr::crossing(sector_frame_df %>% select(xmin, xmax), ref_line_df)
ref_label_df <- tidyr::crossing(
  sector_frame_df %>% transmute(sector, x_label = xmax + sector_gap * 0.32),
  ref_line_df
) %>%
  mutate(label = format(round(y, 2), trim = TRUE, scientific = FALSE))

polar_coord <- if ("coord_radial" %in% getNamespaceExports("ggplot2")) {
  coord_radial(theta = "x", start = 0, end = 2 * pi, clip = "off", inner.radius = 0.34)
} else {
  coord_polar(theta = "x", start = 0, clip = "off")
}

# 2. 绘图：sector -> station_type -> interaction_group
p <- ggplot(df_plot, aes(x = x, y = count, fill = interaction_group)) +
  geom_segment(
    data = ref_segment_df,
    inherit.aes = FALSE,
    aes(x = xmin, xend = xmax, y = y, yend = y),
    color = "grey80",
    linewidth = 0.30
  ) +
  geom_text(
    data = ref_label_df,
    inherit.aes = FALSE,
    aes(x = x_label, y = y, label = label),
    size = 2.45,
    color = "grey40",
    hjust = 0,
    vjust = -0.2
  ) +
  geom_rect(
    data = sector_bg_df,
    inherit.aes = FALSE,
    aes(xmin = xmin, xmax = xmax, ymin = 1, ymax = max_value * 1.08),
    fill = sector_bg_df$bg_fill,
    color = NA,
    alpha = 0.45
  ) +
  geom_rect(
    data = station_bg_df,
    inherit.aes = FALSE,
    aes(xmin = xmin, xmax = xmax, ymin = 1, ymax = max_value * 1.05),
    fill = station_bg_df$bg_fill,
    color = NA,
    alpha = 0.30
  ) +
  geom_rect(
    data = station_frame_df,
    inherit.aes = FALSE,
    aes(xmin = xmin, xmax = xmax, ymin = 1, ymax = max_value * 1.05),
    fill = NA,
    color = "grey65",
    linewidth = 0.34
  ) +
  geom_rect(
    data = sector_frame_df,
    inherit.aes = FALSE,
    aes(xmin = xmin, xmax = xmax, ymin = 1, ymax = max_value * 1.08),
    fill = NA,
    color = "grey35",
    linewidth = 0.7
  ) +
  geom_col(width = bar_width, color = "black", linewidth = 0.18) +
  scale_fill_manual(values = group_palette, breaks = interaction_levels, name = "Interaction group") +
  polar_coord +
  scale_y_log10(expand = expansion(mult = c(0.02, 0.08))) +
  geom_text(
    data = station_label_df,
    aes(x = center, y = max_value * 1.03, label = station_type),
    inherit.aes = FALSE,
    size = 3.1,
    fontface = "bold",
    color = "grey15"
  ) +
  geom_text(
    data = sector_label_df,
    aes(x = center, y = max_value * 1.16, label = sector),
    inherit.aes = FALSE,
    size = 4.2,
    fontface = "bold",
    color = "black"
  ) +
  theme_minimal(base_family = "Times", base_size = 13) +
  theme(
    axis.text = element_blank(),
    axis.title = element_blank(),
    panel.grid = element_blank(),
    legend.position = c(0.5, 0.5),
    legend.justification = c(0.5, 0.5),
    legend.direction = "vertical",
    legend.background = element_rect(fill = scales::alpha("white", 0.78), color = NA),
    legend.key = element_rect(fill = "transparent", color = NA),
    legend.title = element_text(face = "bold", size = 12),
    legend.text = element_text(size = 11),
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16)
  ) +
  labs(title = "Circular Barplot of Dominant Interactions")

print(p)

# 3. 保存为高清矢量图 (用于论文)
ggsave("circular_plot.pdf", width = 10, height = 10)
ggsave("circular_plot.png", width = 10, height = 10, dpi = 320)