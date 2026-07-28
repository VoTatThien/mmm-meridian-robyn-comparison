##############################################################
# robyn_india.R  -  Meta Robyn v3.12.1
# National MMM: BrandA aggregate across 8 Geo (India FMCG)
# 3 lần chạy độc lập (seed = 1, 2, 3)
##############################################################

suppressPackageStartupMessages({
  library(Robyn)
  library(tidyverse)
  library(lubridate)
})

DATA_DIR <- "d:/MARKETING MIX MODELING/DATASET"
OUT_DIR  <- "d:/MARKETING MIX MODELING/DATASET"

# ─── 1. ĐỌC DỮ LIỆU NATIONAL ─────────────────────────────────────
cat("="  , strrep("=", 59), "\n")
cat("ROBYN INDIA — BrandA National Aggregate\n")
cat("=" , strrep("=", 59), "\n")

df_all   <- read_csv(file.path(DATA_DIR, "brandA_national_all.csv"),
                     col_types = cols(Week = col_date()))
df_train <- read_csv(file.path(DATA_DIR, "brandA_national_train.csv"),
                     col_types = cols(Week = col_date()))
df_test  <- read_csv(file.path(DATA_DIR, "brandA_national_test.csv"),
                     col_types = cols(Week = col_date()))

cat(sprintf("Train: %d tuần (%s → %s)\n",
            nrow(df_train), min(df_train$Week), max(df_train$Week)))
cat(sprintf("Test : %d tuần (%s → %s)\n",
            nrow(df_test),  min(df_test$Week),  max(df_test$Week)))

# Định nghĩa biến
MEDIA_VARS    <- c("TV_Impressions", "Digital_Impressions",
                   "Print_Readership", "Radio_Listenership")
MEDIA_LABELS  <- c("TV", "Digital", "Print", "Radio")
PROPHET_VARS  <- c("trend", "season", "holiday")     # Prophet components
CONTROL_VARS  <- c("CPI", "Festival_Index", "Rainfall_Index", "Trade_Spend")

# Đổi tên cột cho dễ đọc
df_robyn <- df_all %>%
  rename(
    date          = Week,
    revenue       = Sales_Value,
    TV            = TV_Impressions,
    Digital       = Digital_Impressions,
    Print         = Print_Readership,
    Radio         = Radio_Listenership
  )

cat(sprintf("Shape: %d rows x %d cols\n", nrow(df_robyn), ncol(df_robyn)))
cat("Cột:", paste(names(df_robyn), collapse=", "), "\n\n")

# ─── 2. ROBYN INPUT SPEC ──────────────────────────────────────────
InputCollect <- robyn_inputs(
  dt_input        = df_robyn,
  dt_holidays     = dt_prophet_holidays,   # built-in holidays
  date_var        = "date",
  dep_var         = "revenue",
  dep_var_type    = "revenue",

  # Prophet decomposition (trend + seasonality + holiday)
  prophet_vars    = PROPHET_VARS,
  prophet_country = "IN",                  # India

  # Paid media channels (impressions)
  paid_media_spends   = c("TV", "Digital", "Print", "Radio"),
  paid_media_vars     = c("TV", "Digital", "Print", "Radio"),

  # Organic / context controls
  organic_vars    = NULL,
  context_vars    = c("CPI", "Festival_Index", "Rainfall_Index", "Trade_Spend"),

  # Adstock: geometric decay
  adstock         = "geometric",

  # Decay theta ranges:
  # Digital: [0.1, 0.4] — hiệu ứng tích lũy ngắn
  # Traditional (TV, Print, Radio): [0.3, 0.8] — tích lũy dài hơn
  hyperparameters = list(
    TV_alphas          = c(0.5, 3.0),
    TV_gammas          = c(0.3, 1.0),
    TV_thetas          = c(0.3, 0.8),

    Digital_alphas     = c(0.5, 3.0),
    Digital_gammas     = c(0.3, 1.0),
    Digital_thetas     = c(0.1, 0.4),

    Print_alphas       = c(0.5, 3.0),
    Print_gammas       = c(0.3, 1.0),
    Print_thetas       = c(0.3, 0.8),

    Radio_alphas       = c(0.5, 3.0),
    Radio_gammas       = c(0.3, 1.0),
    Radio_thetas       = c(0.3, 0.8)
  ),

  # Train period: 80% đầu
  window_start    = min(df_train$Week),
  window_end      = max(df_train$Week)
)

cat("\n✅ InputCollect OK\n")
print(InputCollect)

# ─── 3. CHẠY 3 SEED ───────────────────────────────────────────────
SEEDS       <- c(1, 2, 3)
N_ITER      <- 1000    # số vòng lặp Nevergrad
N_TRIALS    <- 3       # số lần thử mỗi vòng

all_results <- list()

for (seed in SEEDS) {
  cat("\n", strrep("=", 63), "\n")
  cat(sprintf("CHẠY SEED = %d\n", seed))
  cat(strrep("=", 63), "\n")

  set.seed(seed)
  t_start <- proc.time()

  tryCatch({
    # Chạy Robyn
    OutputModels <- robyn_run(
      InputCollect   = InputCollect,
      cores          = 1,             # 1 core để đảm bảo reproducible
      iterations     = N_ITER,
      trials         = N_TRIALS,
      ts_validation  = TRUE,          # time-series validation
      add_penalty_factor = FALSE,
      seed           = seed
    )

    # Collect and calculate Pareto fronts
    OutputCollect <- robyn_outputs(
      InputCollect, OutputModels,
      pareto_fronts = "auto",
      csv_out = "pareto",
      clusters = TRUE,
      export = TRUE,
      plot_folder = "d:/MARKETING MIX MODELING/DATASET"
    )

    # Read pareto_aggregated.csv to find the best model
    pareto_file <- list.files(OutputCollect$plot_folder, pattern = "pareto_aggregated.csv", full.names = TRUE)
    if (length(pareto_file) == 0) {
      stop("pareto_aggregated.csv not found")
    }
    pareto_df <- read.csv(pareto_file[1])
    
    # Chọn mô hình có nrmse thấp nhất
    sol_df <- pareto_df %>%
      arrange(nrmse) %>%
      head(1)
    sel_id <- sol_df$solID[1]
    
    nrmse_val       <- sol_df$nrmse[1]
    decomp_rssd_val <- sol_df$decomp.rssd[1]
    r2_train        <- sol_df$rsq_train[1]
    r2_val          <- sol_df$rsq_val[1]
    r2_test         <- sol_df$rsq_test[1]

    # Decomposition (contribution %)
    decomp_df <- pareto_df %>%
      filter(solID == sel_id) %>%
      select(rn, xDecompPerc)

    # ROI
    roi_df <- pareto_df %>%
      filter(solID == sel_id) %>%
      select(rn, roi_total)

    elapsed <- (proc.time() - t_start)["elapsed"]

    cat(sprintf("  ✅ Seed %d hoàn thành (%.1f phút)\n", seed, elapsed/60))
    cat(sprintf("  NRMSE=%.4f, DECOMP.RSSD=%.4f\n", nrmse_val, decomp_rssd_val))
    cat(sprintf("  R² train=%.4f, val=%.4f, test=%.4f\n", r2_train, r2_val, r2_test))
    cat("  Contribution %:\n")
    print(decomp_df)
    cat("  ROI:\n")
    print(roi_df)

    all_results[[as.character(seed)]] <- list(
      status          = "OK",
      elapsed_min     = elapsed / 60,
      sel_id          = sel_id,
      nrmse           = nrmse_val,
      decomp_rssd     = decomp_rssd_val,
      r2_train        = r2_train,
      r2_val          = r2_val,
      r2_test         = r2_test,
      decomp          = decomp_df,
      roi             = roi_df,
      OutputModels    = OutputModels,
      OutputCollect   = OutputCollect
    )

  }, error = function(e) {
    cat(sprintf("  ❌ Lỗi seed %d: %s\n", seed, conditionMessage(e)))
    all_results[[as.character(seed)]] <<- list(
      status = "ERROR",
      error  = conditionMessage(e)
    )
  })
}

# ─── 4. TỔNG KẾT VÀ LƯU KẾT QUẢ ─────────────────────────────────
cat("\n", strrep("=", 63), "\n")
cat("TỔNG KẾT ROBYN\n")
cat(strrep("=", 63), "\n")

summary_rows <- list()
for (seed in SEEDS) {
  res <- all_results[[as.character(seed)]]
  if (res$status == "OK") {
    cat(sprintf("  Seed %d: ✅ %.1f phút | NRMSE=%.4f | R²_test=%.4f\n",
                seed, res$elapsed_min, res$nrmse, res$r2_test))

    # Thu thập ROI cho từng kênh
    roi_row <- res$roi %>%
      pivot_wider(names_from = rn, values_from = roi_total) %>%
      mutate(seed = seed, status = "OK",
             nrmse = res$nrmse, decomp_rssd = res$decomp_rssd,
             r2_train = res$r2_train, r2_val = res$r2_val, r2_test = res$r2_test)
    summary_rows[[length(summary_rows) + 1]] <- roi_row
  } else {
    cat(sprintf("  Seed %d: ❌ %s\n", seed, res$error))
  }
}

# Tổng hợp ROI across seeds
if (length(summary_rows) > 0) {
  summary_df <- bind_rows(summary_rows)
  write_csv(summary_df, file.path(OUT_DIR, "robyn_roi_summary.csv"))

  # Tính mean ± sd ROI
  media_channels <- c("TV", "Digital", "Print", "Radio")
  cat("\n  ROI Mean ± SD qua 3 seed:\n")
  for (ch in media_channels) {
    if (ch %in% names(summary_df)) {
      vals <- summary_df[[ch]]
      cat(sprintf("    %s: %.4f ± %.4f (CV=%.1f%%)\n",
                  ch, mean(vals, na.rm=T), sd(vals, na.rm=T),
                  sd(vals, na.rm=T)/abs(mean(vals, na.rm=T))*100))
    }
  }

  cat("\n  Contribution % cho seed đầu tiên:\n")
  if (all_results[["1"]]$status == "OK") {
    print(all_results[["1"]]$decomp)
  }

  # Metrics summary
  metrics_df <- summary_df %>%
    select(seed, status, nrmse, decomp_rssd, r2_train, r2_val, r2_test)
  cat("\n  Predictive metrics:\n")
  print(metrics_df)
  write_csv(metrics_df, file.path(OUT_DIR, "robyn_metrics_summary.csv"))
}

cat("\n✅ Robyn hoàn thành!\n")
cat(sprintf("Kết quả lưu tại: %s\n", OUT_DIR))
