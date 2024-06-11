# installing and loading packages 
library(here)
library(tidyverse)

# setting up path
path <- here("data", "country_lifetables", "Mx_5x1")
path

# read in data
countries <- c("AUS", "AUT", "BEL", "BGR", "BLR", "CAN", "CHE", "CZE",
               "DNK", "ESP", "EST", "FIN", "FRATNP", "GBRTENW",
               "GBR_NIR", "GBR_SCO", "GRC", "HUN", "IRL", "ISL",
               "ISR", "ITA", "JPN", "LTU", "LUX", "LVA", "NLD", "NOR",
               "NZL_NM", "POL", "PRT", "RUS", "SVK", "SVN", "SWE", "TWN",
               "UKR", "USA")
country_data <- list()
for (i in 1:length(countries)){
  file <- paste(countries[i], "Mx_5x1.txt", sep = ".")
  country_data[[i]] <- read.table(paste(path, file, sep = "/"),
                                  header = TRUE,
                                  skip = 2,
                                  sep = "",
                                  stringsAsFactors = FALSE)
  country_data[[i]]$Country = countries[i]
}
names(country_data) <- countries

# merge all countries 
all_countries <- do.call(rbind, country_data)

# get gender variable and mort rate variable
all_countries_long <- all_countries %>%
  gather(key = "Gender", value = "Mortality_rate", c("Female", "Male"))

all_countries_long$Total <- NULL

# setting up save_path
save_path <- here("data")
save_path

write.csv(all_countries_long, paste(save_path, "hmd_5yr.csv", sep = "/"), row.names = FALSE)




