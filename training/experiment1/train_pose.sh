#!/usr/bin/env sh
/home/robbie/caffe_train/build/tools/caffe train --solver=pose_solver.prototxt --gpu=$1 --snapshot=/home/robbie/data/Pose_Estimation_Kinematics/training/experiment1model/pose_iter_114000.solverstate 2>&1 | tee /home/robbie/data/Pose_Estimation_Kinematics/outputs/experiment1/output.txt
