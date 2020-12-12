#!/bin/bash

echo "id filter_num1 kernel_size1 filter_num2 kernel_size2 hidden_layer1 hidden_layer2 learning_rate batch_size train_accuracy validation_accuracy time/s" > all_res.txt;

for i in res_*.txt; 
do
	num=$(echo $i | sed 's/[^0-9]*//g') ; 
	hyper=$(cat $i);
	res="${num} ${hyper}";
	echo $res | sed "s/ $//g" >> all_res.txt;
done;
