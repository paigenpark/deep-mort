# installing and loading packages 
library(here)
library(tidyverse)

# setting up path
path <- here("data", "hmd_death_rates", "Mx_1x1")
path

file <- "Mx_1x1.txt"
country_data <- read.table(paste(path, file, sep = "/"),
                                header = TRUE,
                                skip = 2,
                                sep = "",
                                stringsAsFactors = FALSE)

# get gender variable and mort rate variable
all_countries_long <- country_data |>
  pivot_longer(cols = c(Female, Male), 
    names_to = "Gender", values_to = "MortalityRate") |>
  mutate(Gender = case_when(
    Gender == "Female" ~ "f",
    Gender == "Male" ~ "m",
    TRUE ~ Gender  
  ))

all_countries_long$Total <- NULL
head(all_countries_long)

all_countries_long <- all_countries_long |>
  select(PopName, Gender, Year, Age, MortalityRate)
head(all_countries_long)

# setting up save_path
save_path <- here("data")
save_path

write.csv(all_countries_long, paste(save_path, "hmd.csv", sep = "/"), row.names = FALSE)




