# --------------------------------------------- #
# limpieza de datos 
# --------------------------------------------- #
# Infectados nuevos => Infectados acumulados
# --------------------------------------------- #
df <- read.csv("casosnuevosMx.csv")
# sólo quiero ver los de CDMX
# NOTA: El tiempo t0 equivale al 26 de febrero de 2020
cdmx.nuevos_casos <- df[df$nombre == "DISTRITO FEDERAL", 
                        seq(4,104)]
cdmx.nuevos_casos <- unlist(cdmx.nuevos_casos, use.names=FALSE)
# Calcula los infectados acumulados
cdmx.acumulados <- cumsum(cdmx.nuevos_casos)
# Guardamos el vector como un objeto de R
save(cdmx.acumulados, file = "data.Rdata")



