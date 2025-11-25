setwd("~/Documentos/Universidad/Tesis/code")

pob <- read.csv("../data/datos_crudos/modelo_SIS/poblacion_mayor_15_espana.csv")

fit <- lm(population ~ year, data = pob)
summary(fit)

library(car)

# Call:
#   lm(formula = population ~ year, data = pob)
# 
# Residuals:
#   Min      1Q  Median      3Q     Max 
# -216983 -140091   21544   96309  326905 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)
# (Intercept) -6062725   35482995  -0.171    0.868
# year           22669      17618   1.287    0.230
# 
# Residual standard error: 184800 on 9 degrees of freedom
# Multiple R-squared:  0.1554,	Adjusted R-squared:  0.06152 
# F-statistic: 1.655 on 1 and 9 DF,  p-value: 0.2303