#!/bin/bash

echo "" > all_res.txt;

for i in res_*.txt; 
do
	num=$(echo $i | sed 's/[^0-9]*//g') ; 
	hyper=$(cat $i);
	res="${num}\t${hyper}";
	echo -e  "$res" >> all_res.txt;
	done;
