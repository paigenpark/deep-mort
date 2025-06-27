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


# --------------------------
# 1. Parameters
# --------------------------
forecast_years    <- 2006:2015
n_iter            <- 1

# --------------------------
# 2. Load raw training data
# --------------------------
country_training <- read.table("../../data/country_training.txt", header = FALSE,
                               col.names = c("Country", "Gender", "Year", "Age", "Rate"))

country_training <- country_training |>
  arrange(Country, Gender, Year, Age)

fifty_to_1960 = c()
greater_than_1960 = c()
years_available = list()  

for (country in unique(country_training$Country)) {
  one_country = country_training |>
    filter(Country == country) 
  
  print(country)
  print(nrow(one_country))
  
  year_min = min(one_country$Year, na.rm = TRUE)
  
  years_available[[country]] = year_min 
  
  if (year_min > 1950 & year_min <= 1960) {
    fifty_to_1960 = c(fifty_to_1960, country)
  }
  if (year_min > 1960) {
    greater_than_1960 = c(greater_than_1960, country)
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
# 3. Prepare the three datasets
# --------------------------

# 3a) FULL‑HISTORY set (problem countries removed)
excluded_countries <- c(fifty_to_1960, greater_than_1960)
full_hist_df <- country_training %>%
  filter(!Country %in% excluded_countries)
full_demog_list   <- build_demog_list(full_hist_df, ages_all)
combined_full     <- combine_demog_list(full_demog_list)

# 3b) RESTRICTED‑YEARS set including only countries with data back to at least 1960
#     1) find the intersection of years that every country–gender combo possesses
common_years <- country_training %>%
  filter(!Country %in% greater_than_1960) %>%
  group_by(Country, Gender) %>%
  summarise(yrs = list(unique(Year)), .groups = "drop") %>%
  pull(yrs) %>%
  reduce(intersect)

df_1960      <- country_training %>% filter(Year %in% common_years)
demog_list_1960 <- build_demog_list(df_1960, ages_all)
combined_restrict_1960    <- combine_demog_list(demog_list_1960)

# 3c) RESTRICTED‑YEARS set for all countries
#     1) find the intersection of years that every country–gender combo possesses
common_years <- country_training %>%
  group_by(Country, Gender) %>%
  summarise(yrs = list(unique(Year)), .groups = "drop") %>%
  pull(yrs) %>%
  reduce(intersect)

df_after_1960      <- country_training %>% filter(Year %in% common_years)
demog_list_after_1960  <- build_demog_list(df_after_1960, ages_all)
combined_restrict_after_1960      <- combine_demog_list(demog_list_after_1960)

# --------------------------
# 4. Fit, forecast, and merge (repeat n_iter times)
# --------------------------
for (iter in seq_len(n_iter)) {
  
  # 4a) Coherent fit on full‑history subset
  set.seed(1000 + iter)             # ensure different SVD starts each loop
  fit_full <- coherentfdm(combined_full)
  fc_full  <- forecast(fit_full, h = length(forecast_years))
 
  
  # 4b) Coherent fit on restricted years (data back to 1960)
  set.seed(2000 + iter)
  fit_restrict_1960 <- coherentfdm(combined_restrict_1960)
  fc_restrict_1960  <- forecast(fit_restrict_1960, h = length(forecast_years))
  
  # 4c) Coherent fit on restricted years (data back to 1960)
  set.seed(2000 + iter)
  fit_restrict_after_1960 <- coherentfdm(combined_restrict_after_1960)
  fc_restrict_after_1960  <- forecast(fit_restrict_after_1960, h = length(forecast_years))
  
  # 4d) Build a unified list of forecasts using only the **population-level** labels
  pop_labels <- names(demog_list_after_1960)  # e.g., "16_1", "16_2", ...
  
  fc_mixed_rates <- map(pop_labels, function(label) {
    country_id <- str_split(label, "_", simplify = TRUE)[1]
    
    if (country_id %in% greater_than_1960) {
      fc_restrict_after_1960[[label]]$rate[[1]]
    } else if (country_id %in% fifty_to_1960) {
      fc_restrict_1960[[label]]$rate[[1]]
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
  

  out_path <- glue("../../data/coherent_forecast_test_{iter}.csv")
  write.table(final_df, out_path, sep = ",", col.names = FALSE, row.names = FALSE)
  
  message(glue("Iteration {iter} complete – saved to {out_path}"))
}




