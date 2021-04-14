#!/bin/bash


echo "copying data to cache"
cp -r /scratch/jane-doe-framework/datasets/ml-20m/ /ssd/ml-20m

echo "done copying data to cache"
ls -la /ssd