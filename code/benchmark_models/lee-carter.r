library(demography)
library(tidyverse)
library(reshape2)

# sets working directory to the location of this script
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

country_training <- read.table("../../data/country_training.txt", header = FALSE)
countries <- unique(country_training[,1])
genders <- unique(country_training[,2])
years <- unique(country_training[,3])
ages <- unique(country_training[,4])
forecasted_years <- 2006:2015
colnames(country_training) <- c('Country', 'Gender', 'Year', 'Age', 'Rate')

fitted_results <- list()
forecasted_results <- list()

for (r in 1:5) {
  for (i in countries) {
    for (j in genders) {
      filtered <- country_training |>  
        filter(country_training[,1] == i & country_training[,2] == j) 
      
      mx_mat <- unlist(lapply(years, function(i) filtered[filtered[,3] == i, 5]))
      mx_mat <- matrix(mx_mat, nrow = length(ages), ncol = length(years), byrow = FALSE)
      colnames(mx_mat) <- years
      mx_mat[is.na(mx_mat)] <- 1e-6
      mx_mat[mx_mat == 0] <- 1e-6
      
      Ext <- matrix(1, nrow = length(ages), ncol = length(years))
      
      data <- demogdata(
        data = mx_mat,
        pop = Ext,
        ages = ages,
        years = years,
        type = "mortality",
        label = i,
        name = j
      )
      
      lc_output <- lca(data,
                       years = years,
                       ages = ages)
      
      fitted <- exp(lc_output$fitted$y)
      df_fitted <- as.data.frame(fitted)
      df_fitted$age <- ages
      df_fitted_long <- melt(df_fitted, id.vars = "age", 
                             variable.name = "year",
                             value.name = "rate")
      df_fitted_long$year <- rep(years, each = length(ages))
      df_fitted_long$country <- i
      df_fitted_long$gender <- j
      fitted_results[[paste(i, j, sep = "_")]] <- df_fitted_long
        
      forecasted <- forecast(lc_output, h=10) 
      forecasted_rates <- do.call(cbind, forecasted$rate[1])
      df_forecasted <- as.data.frame(forecasted_rates)
      df_forecasted$age <- ages
      df_forecasted_long <- melt(df_forecasted, id.vars = "age", 
                             variable.name = "year",
                             value.name = "rate")
      df_forecasted_long$year <- rep(forecasted_years, each = length(ages))
      df_forecasted_long$country <- i
      df_forecasted_long$gender <- j
      forecasted_results[[paste(i, j, sep = "_")]] <- df_forecasted_long
    }
  }
  
  final_fitted_df <- bind_rows(fitted_results)
  final_fitted_df <- final_fitted_df |>
    select(country, gender, year, age, rate)
  final_forecasted_df <- bind_rows(forecasted_results)
  final_forecasted_df <- final_forecasted_df |>
    select(country, gender, year, age, rate)  
  
  # View structure of final datasets
  head(final_fitted_df)
  head(final_forecasted_df)
  
  if (!"glue" %in% installed.packages()) install.packages("glue")
  library(glue)
  
  # uncomment to re-save prediction files
  #write.table(final_fitted_df, "../../data/lc_fitted_all.csv", sep=",", col.names = FALSE,
            #  row.names = FALSE)
}


