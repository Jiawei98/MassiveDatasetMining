#!/bin/bash

mkdir -p images_tgz

for num in $(seq -w 1 58); do
	tar czf images_tgz/images_${num}.tgz images/${num}???.jpg
	echo $num
done
tar czf images_tgz/images_59.tgz images/59???.jpg images/60000.jpg
