import os
import time

# should restart training from snap
restart_training = True

# GPU
used_gpu = '0,1,2,3'

# caffe exec
caffe_exec_path = '/tanfulun/workspaces/Project/caffe-cpm/caffe/build/tools/caffe'

# project dir
project_dir = '/tanfulun/workspaces/Project/clothing_key_point_detection/convolutional-pose-machines-release/'

# solver
solver_path = os.path.join(project_dir, 'training/prototxt/DeepFashion_Landmark/pose_solver.prototxt')

#------------------
stime_str = time.ctime()
stime = time.time()
print(stime_str)

if not restart_training:
    os.system('nohup %s train --solver=%s -gpu %s >caffe_log.log 2>&1'%(caffe_exec_path,solver_path,used_gpu))
else:
    iteration = 6000
    snapshot = project_dir + '/training/prototxt/DeepFashion_Landmark/caffemodel/landmark_iter_%s.solverstate'%iteration
    os.system('nohup %s train --solver=%s --snapshot=%s -gpu %s >caffe_log_%s.log 2>&1'%(caffe_exec_path,solver_path,snapshot,used_gpu,iteration))

#end-training-time
etime_str = time.ctime()
etime = time.time()
print(etime_str)

# Training Time
dur_sec = etime-stime

hour = int(dur_sec/3600)
minute = int( (dur_sec-3600*hour)/60 )

print ('%d h, %d m'%(hour,minute))

