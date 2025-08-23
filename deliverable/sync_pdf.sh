#!/bin/bash

# Nombre del PDF
PDF="main.pdf"

# Ruta local
LOCAL_PATH="./$PDF"
# Ruta en tu Drive montado (ajustado)
DRIVE_DIR="/home/edtrelo/Documentos/Google Drive/edtrelo@ciencias.unam.mx/[01] Tesis"
DRIVE_PATH="$DRIVE_DIR/$PDF"

# Crear carpeta en Drive si no existe
mkdir -p "$DRIVE_DIR"

# Verificar si el PDF cambió
if [ ! -f "$DRIVE_PATH" ] || ! cmp -s "$LOCAL_PATH" "$DRIVE_PATH"; then
    cp "$LOCAL_PATH" "$DRIVE_PATH"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - PDF actualizado en Drive ✅"
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') - PDF sin cambios, no se copia"
fi


