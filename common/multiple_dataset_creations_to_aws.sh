#!/usr/bin/env bash

start=`date +%s`

# setting what variable you want to loop through (segment length, hop length, overlap)
for value in 3.0
do
    # timing the script
    startloop=`date +%s`

    # run the proprocessing script
    python dataset_creation.py -seg $value -hop 0.5 -ovlap 0.5

    # moving to the data directory to copy over to AWS
    cd data/dwt_df_parquet/ 

    folder=seg_"$value"_hop_0.5_ovlap_0.5

    # creating AWS bucket
    aws s3api put-object --bucket spotify-podcasts-data --key $folder/
    # putting the files into AWS bucket
    # https://bobbyhadz.com/blog/aws-s3-copy-local-folder-to-bucket
    aws s3 sync $folder  s3://spotify-podcasts-data/$folder/

    runtime=$((($(date +%s)-$startloop)/60))
    echo "preprocessing.sh for $folder completed in $runtime minutes"

    # reset back to top-level directory
    cd ../../

done

runtime=$((($(date +%s)-$start)/60))
echo "Completed fully: preprocessing.sh done in $runtime minutes"

