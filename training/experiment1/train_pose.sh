#!/usr/bin/env sh
/home/robbie/caffe_train/build/tools/caffe train --solver=pose_solver.prototxt --gpu=$1 --weights=/home/robbie/data/model/alexnet/bvlc_alexnet.caffemodel 2>&1 | tee /home/robbie/data/Pose_Estimation_Kinematics/outputs/experiment1/output.txt
