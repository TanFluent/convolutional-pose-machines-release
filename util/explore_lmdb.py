import caffe
import lmdb
import numpy as np
import cv2
from caffe.proto import caffe_pb2
import pdb

lmdb_data_dir = '/home/tfl/workspace/project/convolutional-pose-machines-release/training/lmdb/FLIC/'
#lmdb_data_dir = '/home/tfl/workspace/project/convolutional-pose-machines-release/dataset/DeepfashionLandmark/lmdb/test100'

lmdb_env = lmdb.open(lmdb_data_dir)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe_pb2.Datum()

#pdb.set_trace()

for key, value in lmdb_cursor:
    datum.ParseFromString(value)

    label = datum.label
    data = caffe.io.datum_to_array(datum)

    #pdb.set_trace()

    if data.shape[0]==4:
        # split "image" and "annotation info"
        image_data = data[0:3]
        anno_info = data[3]
    else:
        image_data = data

    #pdb.set_trace()

    #CxHxW to HxWxC in cv2
    image = np.transpose(image_data, (1,2,0))
    cv2.imshow('cv2', image)

    print('{},{}'.format(key, label))

    k = cv2.waitKey(0)
    if k == ord('q'):
        cv2.destroyAllWindows()
        break
    else:
        cv2.destroyAllWindows()
        continue

