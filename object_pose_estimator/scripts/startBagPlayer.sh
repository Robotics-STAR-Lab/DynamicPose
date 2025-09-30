#!/bin/bash
# run_bag_player.sh

BAG_FILE=$1   # 第一个参数是 bag 文件路径
echo "Launching bag player with bag file: $BAG_FILE"

# 激活环境（如果需要）
# source /opt/ros/noetic/setup.bash

BASE_ENV_PATH=$(conda info | grep "base environment" | awk '{print $4}')
PYTHON=$BASE_ENV_PATH/envs/dynamicpose/bin/python

# 调用 python 脚本
$PYTHON $(rospack find object_pose_estimator)/scripts/bag_player.py --bag $BAG_FILE
