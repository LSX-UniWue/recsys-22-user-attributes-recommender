#!/bin/bash

echo "copying data to cache"
cp -r /scratch/jane-doe-framework/datasets/conventions/ml-20m_3_0_0/ /ssd/ml-20m

echo "done copying data to cache"
ls -la /ssd
ls -la /ssd/ml-20m