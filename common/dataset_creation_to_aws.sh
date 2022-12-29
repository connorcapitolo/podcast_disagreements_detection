#!/usr/bin/env bash

# timing the script
start=`date +%s`

# run the proprocessing script
python dataset_creation.py -seg $1 -hop $2 -ovlap $3

# moving to the data directory to copy over to AWS
cd data/dwt_df_parquet/ 

folder=seg_$1_hop_$2_ovlap_$3

# creating AWS bucket
aws s3api put-object --bucket spotify-podcasts-data --key $folder/

# putting the files into AWS bucket
# https://bobbyhadz.com/blog/aws-s3-copy-local-folder-to-bucket
aws s3 sync $folder  s3://spotify-podcasts-data/$folder/

# how long it took to run
end=`date +%s`
runtime=$((end-start))
echo "preprocessing.sh completed in $runtime seconds"
