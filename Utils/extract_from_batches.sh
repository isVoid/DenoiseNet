#!/bin/bash

root="/media/michael/Work Drive 1/Pairs_TIFF/"
declare -a clean=("b3clean" "b4clean" "b5Clean" "Cleanb6" "Cleanb7" "Cleanb8" "Cleanb9" "Cleanb10" "Cleanb12" "Cleanb13")
declare -a noisy=("b3noisy" "b4noisy" "b5Noisy" "Noisyb6" "Noisyb7" "Noisyb8" "Noisyb9" "Noisyb10" "Noisyb12" "Noisyb13")

arraylength=${#clean[@]}

for ((i=0; i < $arraylength; i++));
do
  c="\""$root${clean[i]}"/\""
  echo $c
  n="\""$root${noisy[i]}"/\""
  echo $n

  echo python ./Utils/extract_patches.py $n $c
  eval python ./Utils/extract_patches.py $n $c
done
