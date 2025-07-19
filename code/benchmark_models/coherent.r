# Coherent Two-Stage Forecast
# ---
# Goal: 1) Fit a coherent product–ratio model on the **maximal** set of
#        countries (problem countries removed), using their full histories.
#        2) Fit a second coherent model restricted to the common time window
#        across **all** countries (including the five problem ones).
#        3) For forecast years 2006–2015, export a single CSV where
#        predictions for the five problem countries come from the restricted
#        model and all others come from the full‐history model.
#        4) Repeat the whole process `n_iter` times.
#
# Note: The forecast() call below often prints
# 
#   Error in stats::arima(x = …): non-stationary AR part from CSS
# 
# ▸ Why it appears  
#   auto.arima() searches many candidate ARIMA orders.  Each one is first
#   estimated with conditional-sum-of-squares (“CSS”).  If the provisional
#   AR coefficients are outside the stationarity region, stats::arima()
#   throws the message *before* control returns to R.  The call is wrapped
#   in try(), so execution continues.
#
# ▸ Why it is harmless  
#   The candidate that fails is discarded (its IC is set to Inf) and the
#   search moves on.  The winning order is ultimately *re-fitted* with full
#   maximum likelihood, so the final model is guaranteed stationary.
#
# Source: forecast::auto.arima() and helper myarima()  
# https://github.com/robjhyndman/forecast/blob/master/R/newarima2.R

# --------------------------------------------------------------------------------
library(demography)
library(tidyverse)
library(reshape2)
library(glue)
library(here)



# --------------------------
# 1. Parameters and paths
# --------------------------
forecast_years    <- 2006:2015
n_iter            <- 5
path <- here("data")

# --------------------------
# 2. Load raw training data
# --------------------------
file <- "country_training.txt"
country_training <- read.table(paste(path, file, sep = "/"), header = FALSE,
                               col.names = c("Country", "Gender", "Year", "Age", "Rate"))


country_training <- country_training |>
  arrange(Country, Gender, Year, Age)

excluded_countries = c()

for (country in unique(country_training$Country)) {
  one_country = country_training |>
    filter(Country == country) 
  
  year_min = min(one_country$Year, na.rm = TRUE)
  
  if (year_min > 1959) {
    excluded_countries = c(excluded_countries, country)
  }
}


ages_all <- sort(unique(country_training$Age))

# build a list of demogdata objects (one per country–gender combo)
build_demog_list <- function(df, ages_vec) {
  data_objects <- list()
  for (c in unique(df$Country)) {
    for (g in unique(df$Gender)) {
      sub <- df %>% filter(Country == c, Gender == g)
      years_i <- sort(unique(sub$Year))
      
      mx_df <- sub %>%
        pivot_wider(names_from = "Year", values_from = "Rate") %>%
        select(-Country, -Gender, -Age)
      mx_mat <- as.matrix(mx_df)
      mx_mat[mx_mat == 0 | is.na(mx_mat)] <- 9e-06
      
      Ext <- matrix(1, nrow = nrow(mx_mat), ncol = ncol(mx_mat))
      
      data <- demogdata(
        data  = mx_mat,
        pop   = Ext,
        ages  = ages_vec,
        years = years_i,
        type  = "mortality",
        label = c,
        name  = g
      )
      list_name <- glue("{c}_{g}")
      data_objects[[list_name]] <- data
    }
  }
  data_objects
}

# collapse a list of demogdata objects into a single combined object
combine_demog_list <- function(dlist) {
  template <- dlist[[1]]
  template$rate <- lapply(dlist, function(x) x$rate[[1]])
  template$pop  <- lapply(dlist, function(x) x$pop[[1]])
  template
}

# --------------------------
# 3. Prepare the datasets
# --------------------------

# 3a) FULL‑HISTORY set (problem countries removed)
full_hist_df <- country_training %>%
  filter(!Country %in% excluded_countries)
full_demog_list   <- build_demog_list(full_hist_df, ages_all)
combined_full     <- combine_demog_list(full_demog_list)

# 3c) RESTRICTED‑YEARS set for all countries (except Germany - which was causing problems)
# find the intersection of years that every country–gender combo possesses
germany <- c(58, 59)
common_years <- country_training %>%
  group_by(Country, Gender) %>%
  summarise(yrs = list(unique(Year)), .groups = "drop") %>%
  pull(yrs) %>%
  reduce(intersect)

df_restricted      <- country_training %>% filter(Year %in% common_years,
                                                  !Country %in% germany)
demog_list_restricted  <- build_demog_list(df_restricted, ages_all)
combined_restricted      <- combine_demog_list(demog_list_restricted)




# --------------------------
# 4. Fit, forecast, and merge (repeat n_iter times)
# --------------------------
for (iter in seq_len(n_iter)) {
  set.seed(iter)
  # 4a) Coherent fit on full‑history subset
  fit_full <- coherentfdm(combined_full)
  fc_full  <- forecast(fit_full, h = length(forecast_years))
  
  # 4c) Coherent fit on restricted years (data back to 1960)
  fit_restrict <- coherentfdm(combined_restricted)
  fc_restrict  <- forecast(fit_restrict, h = length(forecast_years))
  
  # 4d) Build a unified list of forecasts using only the **population-level** labels
  pop_labels <- union(names(full_demog_list),   # everything in the full model
                      names(demog_list_restricted))  # e.g., "16_1", "16_2", ...
  
  fc_mixed_rates <- map(pop_labels, function(label) {
    country_id <- as.integer(str_split(label, "_", simplify = TRUE)[1])
    if (country_id %in% excluded_countries) {
      fc_restrict[[label]]$rate[[1]]
    } else {
      fc_full[[label]]$rate[[1]]
    }
  })
  names(fc_mixed_rates) <- pop_labels
  
  
  # 4d) Convert to long data frame
  forecasted_results <- imap(fc_mixed_rates, function(mat, label) {
    df <- as.data.frame(mat)
    df$age <- ages_all
    long_df <- melt(df, id.vars = "age", variable.name = "year", value.name = "rate")
    long_df$year <- rep(forecast_years, each = length(ages_all))
    label_split <- str_split(label, "_", simplify = TRUE)
    long_df$country <- as.numeric(label_split[1])
    long_df$gender  <- as.numeric(label_split[2])
    long_df %>% select(country, gender, year, age, rate)
  })
  
  final_df <- bind_rows(forecasted_results)
  
  out_file <- glue("coherent_forecast_{iter}.csv")
  write.table(final_df, paste(path, out_file, sep = "/"), sep = ",", col.names = FALSE, row.names = FALSE)
  
  print(glue("Iteration {iter} complete – saved to {out_file}"))
}

