#! /bin/bash
Rscript scaler_flights.R
awk -F" " '{print "1",$0}' X_train.txt > X_ent.txt
rm X_train.txt

for i in {1..39}
do
   echo 0.1
done > b_bh.txt

awk -F" " '{print "1",$0}' X_val.txt > X_valida.txt
rm X_val.txt
