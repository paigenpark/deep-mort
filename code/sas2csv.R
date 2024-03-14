options(stringsAsFactors=F, scipen = 999)

pkg = 'Hmisc'
if (!require(pkg, character.only = TRUE)) {
  install.packages(pkg)
  library(pkg, character.only = TRUE)
}

setwd()

brfss <- sasxport.get("BRFSS_2015.XPT")
write.csv(brfss, file = "brfss2015.csv")

brfss <- sasxport.get("BRFSS_2016.XPT")
write.csv(brfss, file = "brfss2016.csv")

brfss <- sasxport.get("BRFSS_2017.XPT")
write.csv(brfss, file = "brfss2017.csv")

brfss <- sasxport.get("BRFSS_2018.XPT")
write.csv(brfss, file = "brfss2018.csv")

brfss <- sasxport.get("BRFSS_2019.XPT")
write.csv(brfss, file = "brfss2019.csv")

brfss <- sasxport.get("BRFSS_2020.XPT")
write.csv(brfss, file = "brfss2020.csv")

brfss <- sasxport.get("BRFSS_2021.XPT")
write.csv(brfss, file = "brfss2021.csv")

brfss <- sasxport.get("BRFSS_2022.XPT")
write.csv(brfss, file = "brfss2022.csv")
