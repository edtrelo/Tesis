# --------------------------------------------- #
# limpieza de datos 
# --------------------------------------------- #
# Infectados nuevos => Infectados acumulados
# --------------------------------------------- #
setwd("~/Documentos/Universidad/Tesis/code")
df <- read.csv("Casos_Diarios_Municipio_Confirmados_20230625.csv")
# sólo quiero ver los de CDMX
# NOTA: El tiempo t0 equivale al 26 de febrero de 2020
cdmx.nuevos_casos <- df[df$nombre == "Merida", 
                        seq(4,104)]
cdmx.nuevos_casos <- unlist(cdmx.nuevos_casos, use.names=FALSE)
# Calcula el número de infectados en los últimos 14 días
cdmx.acum14 <- rep(0, length(cdmx.nuevos_casos))
for(i in 1:length(cdmx.nuevos_casos)){
  # sumar todos los días hasta 7 previos (o menos si no hay suficientes)
  start <- max(1, i-6)  # i-6 para incluir hasta 7 días
  cdmx.acum14[i] <- sum(cdmx.nuevos_casos[start:i])
}
# Calcula los infectados acumulados
cdmx.acumulados <- cumsum(cdmx.nuevos_casos)
# Guardamos el vector como un objeto de R
save(cdmx.nuevos_casos, 
     cdmx.acum14, 
     cdmx.acumulados, file = "data.Rdata")

