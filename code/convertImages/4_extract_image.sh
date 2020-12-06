for tgzfile in $(ls images_*_gray.tgz); do
	tar -xzf $tgzfile -C images_compressed_tgz --strip-components 1 $(echo $tgzfile | cut -d. -f1)
done

