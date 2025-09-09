#!/bin/bash
echo "Pod gotowy! Możesz się połączyć przez SSH i załadować dataset..."
# Pętla keep-alive, żeby kontener nie zamknął się od razu
while true; do sleep 1000; done
