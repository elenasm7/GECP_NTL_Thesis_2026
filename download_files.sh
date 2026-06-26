#!/bin/bash

years=(
    "2015"
    "2016"
    "2017"
)

tiles=(
    "h29v06" 
    "h29v07"
    "h29v08"
    "h30v06"
    "h30v07"
    "h30v08"
)

server="esegrestmorais@snellius.surf.nl:/gpfs/work4/0/FWC2/MYRIAD/data/NTL_data/Processed_data/Black_Marble_processed/corrected_NTL/"
local_destination="data/tiles/"

for y in "${years[@]}"; do
    for t in "${tiles[@]}"; do
        file="${t}_${y}.nc"
        file_path="${server}${y}/nc/${file}"
        local_file_path="${local_destination}${y}/"
        echo "Downloading $file..."
        scp ${file_path} ${local_file_path}
    done
done