# ===========================================================
# THÍ NGHIỆM ROBYN - 3 SEED
# Output: 3 file CSV để phân tích
# Thời gian dự kiến: ~30-60 phút
# ===========================================================

# --- 0. Setup ---
library(Robyn)
library(reticulate)
library(dplyr)
use_virtualenv("r-reticulate", required = TRUE)

# --- 1. Tạo thư mục output ---
output_base_dir <- "./robyn_3seed_results"
dir.create(output_base_dir, showWarnings = FALSE)

# --- 2. Load dữ liệu ---
data("dt_simulated_weekly")
data("dt_prophet_holidays")

media_vars <- c("tv_S", "ooh_S", "print_S", "facebook_S", "search_S")
organic_vars <- "newsletter"

# --- 3. Tự động sinh hyperparameters ---
required_names <- hyper_names(
  adstock = "geometric",
  all_media = c(media_vars, organic_vars)
)

digital_channels <- c("facebook_S", "search_S", "newsletter")

hyperparameters <- list()
for (name in required_names) {
  if (grepl("_alphas$", name)) {
    hyperparameters[[name]] <- c(0.5, 3)
  } else if (grepl("_gammas$", name)) {
    hyperparameters[[name]] <- c(0.3, 1)
  } else if (grepl("_thetas$", name)) {
    channel <- gsub("_thetas$", "", name)
    if (channel %in% digital_channels) {
      hyperparameters[[name]] <- c(0.1, 0.4)
    } else {
      hyperparameters[[name]] <- c(0.3, 0.8)
    }
  }
}
hyperparameters[["train_size"]] <- c(0.5, 0.8)

stopifnot(length(hyperparameters) == 19)
cat(">>> Hyperparameters: 19 phần tử OK\n")

# --- 4. Tạo InputCollect ---
InputCollect <- robyn_inputs(
  dt_input = dt_simulated_weekly,
  dt_holidays = dt_prophet_holidays,
  date_var = "DATE",
  dep_var = "revenue",
  dep_var_type = "revenue",
  prophet_vars = c("trend", "season", "holiday"),
  prophet_country = "DE",
  context_vars = c("competitor_sales_B", "events"),
  paid_media_spends = media_vars,
  paid_media_vars = media_vars,
  organic_vars = organic_vars,
  factor_vars = "events",
  window_start = "2016-11-23",
  window_end = "2018-08-22",
  adstock = "geometric",
  hyperparameters = hyperparameters,
  calibration_input = NULL
)
cat(">>> InputCollect đã tạo\n")

# --- 5. Hàm chạy 1 seed ---
run_one_seed <- function(seed_value) {
  cat(sprintf("\n========================================\n"))
  cat(sprintf(">>> SEED %d - bắt đầu lúc %s\n", 
              seed_value, format(Sys.time(), "%H:%M:%S")))
  cat(sprintf("========================================\n"))
  
  seed_start <- Sys.time()
  
  OutputModels <- robyn_run(
    InputCollect = InputCollect,
    cores = 8,
    iterations = 2000,
    trials = 5,
    ts_validation = TRUE,
    add_penalty_factor = FALSE,
    seed = seed_value,
    nevergrad_algo = "TwoPointsDE"
  )
  
  plot_folder_abs <- normalizePath(
    file.path(output_base_dir, paste0("seed_", seed_value)),
    mustWork = FALSE
  )
  dir.create(plot_folder_abs, showWarnings = FALSE, recursive = TRUE)
  
  OutputCollect <- robyn_outputs(
    InputCollect, OutputModels,
    pareto_fronts = "auto",
    csv_out = "pareto",
    clusters = TRUE,
    plot_pareto = FALSE,
    plot_folder = plot_folder_abs,
    export = TRUE
  )
  
  pareto_models <- OutputCollect$resultHypParam %>%
    filter(robynPareto == 1) %>%
    arrange(nrmse)
  
  best_id <- pareto_models$solID[1]
  
  channel_data <- OutputCollect$xDecompAgg %>%
    filter(solID == best_id, rn %in% media_vars) %>%
    select(rn, xDecompPerc, roi_total)
  
  elapsed_min <- as.numeric(difftime(Sys.time(), seed_start, units = "mins"))
  
  list(
    seed = seed_value,
    best_id = best_id,
    nrmse = pareto_models$nrmse[1],
    decomp_rssd = pareto_models$decomp.rssd[1],
    rsq_train = pareto_models$rsq_train[1],
    rsq_val = pareto_models$rsq_val[1],
    rsq_test = pareto_models$rsq_test[1],
    n_pareto = nrow(pareto_models),
    channel_data = channel_data,
    time_min = round(elapsed_min, 1)
  )
}

# --- 6. Chạy 3 seed với incremental save ---
all_results <- list()
total_start <- Sys.time()

for (s in 1:3) {
  tryCatch({
    all_results[[s]] <- run_one_seed(s)
    
    total_min <- as.numeric(difftime(Sys.time(), total_start, units = "mins"))
    cat(sprintf(">>> Seed %d xong - %.1f phút | Tổng: %.1f phút\n",
                s, all_results[[s]]$time_min, total_min))
    
    # SAVE NGAY sau mỗi seed
    saveRDS(all_results, file.path(output_base_dir, "all_robyn_results.rds"))
    cat(">>> Đã save (an toàn nếu treo)\n")
    
  }, error = function(e) {
    cat(sprintf(">>> LỖI seed %d: %s\n", s, e$message))
    all_results[[s]] <- NULL
  })
}

# --- 7. Lưu kết quả thành 3 file CSV ---
summary_df <- do.call(rbind, lapply(all_results, function(r) {
  if (is.null(r)) return(NULL)
  data.frame(
    seed = r$seed,
    nrmse = r$nrmse,
    decomp_rssd = r$decomp_rssd,
    rsq_train = r$rsq_train,
    rsq_val = r$rsq_val,
    rsq_test = r$rsq_test,
    n_pareto = r$n_pareto,
    time_min = r$time_min
  )
}))
write.csv(summary_df, 
          file.path(output_base_dir, "robyn_summary.csv"),
          row.names = FALSE)

contrib_df <- do.call(rbind, lapply(all_results, function(r) {
  if (is.null(r)) return(NULL)
  data.frame(
    seed = r$seed,
    channel = r$channel_data$rn,
    contribution_pct = r$channel_data$xDecompPerc * 100
  )
}))
write.csv(contrib_df,
          file.path(output_base_dir, "robyn_contributions.csv"),
          row.names = FALSE)

roi_df <- do.call(rbind, lapply(all_results, function(r) {
  if (is.null(r)) return(NULL)
  data.frame(
    seed = r$seed,
    channel = r$channel_data$rn,
    roi = r$channel_data$roi_total
  )
}))
write.csv(roi_df,
          file.path(output_base_dir, "robyn_roi.csv"),
          row.names = FALSE)

cat("\n\n=== HOÀN THÀNH ROBYN 3 SEED ===\n")
cat(sprintf("Tổng thời gian: %.1f phút\n", total_min))
cat("Files đã lưu trong:", output_base_dir, "\n")
cat("- robyn_summary.csv\n")
cat("- robyn_contributions.csv\n")
cat("- robyn_roi.csv\n")