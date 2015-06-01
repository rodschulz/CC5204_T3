#!/bin/bash
for i in `seq 1 5`;
do
	for j in `seq 1 5`;
	do
	rm ../config/config
	rm -r ../images/cat/sample
	rm -r ../images/dog/sample
	rm -r ../images/cow/sample
	s=0.$i
	c=$j
	t=100
	ans=$((c*t))
	echo "sampleImageSet true" >> ../config/config
	echo "sampleSize $s" >> ../config/config
	echo "codebookClusters $ans" >> ../config/config
	echo "RUNNING FOR $s SAMPLE SIZE WITH $ans CLUSTERS"
	(cd ../build/; ./T3 ../images/)
	done
done  
