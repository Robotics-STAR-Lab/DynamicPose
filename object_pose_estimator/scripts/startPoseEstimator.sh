#!/bin/bash
# You should change the path to your own python path
BASE_ENV_PATH=$(conda info | grep "base environment" | awk '{print $4}')
PYTHON=$BASE_ENV_PATH/envs/d6d/bin/python

# $PYTHON --version # ensure the right python version is used
pe_path=$(rospack find object_pose_estimator)
# echo "Package path: $pe_path"
cd $pe_path/scripts
$PYTHON pose_estimator.py
