#!/bin/bash

for image_path in $(ls images/????.jpg); do
	image_fn=$(echo $image_path | cut -d/ -f2)
	mv $image_path images/0$image_fn
done;
