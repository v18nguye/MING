#!/bin/bash 

while getopts t: flag
do
    case "${flag}" in
        t) type=${OPTARG};;
    esac
done

if [ "$type" = "ckpt" ]
    then 
    echo "downloading ckpts!" 

    curl -L -o ckpts.zip https://drive.switch.ch/index.php/s/iV6u5KriKmxftsx/download
    unzip ckpts.zip
    rm ckpts.zip
else
    echo "downloading data!" 

    cd ming
    curl -L -o data.zip https://drive.switch.ch/index.php/s/wHWDy7nyf3cvYBW/download
    unzip data.zip
    rm data.zip

    curl -L -o processed_data.zip https://drive.switch.ch/index.php/s/YavAMhEU6bJW0ud/download
    unzip processed_data.zip
    rm processed_data.zip
    cd ..
fi