import os
import numpy as np
import struct
import cv2
import lmdb
import caffe
import pdb


class DeepFashionLandmark:

    def __init__(self, dataset_dir, save_dir):

        # --Dataset Info
        self._cloth_type = ['None', 'upper-body', 'lower-body', 'full-body']
        self._pose_type = ['None', 'normal pose', 'medium pose', 'large pose', 'medium zoom-in', 'large zoom-in']
        self._visibility_type = ['visible', 'invisible', 'truncated']
        self._dataset_name = 'DeepFashion_Landmark'
        self._num_of_max_landmark = 18
        self._image_min_width = 18 * 4

        # --Dataset Dir
        self._dataset_dir = dataset_dir
        if not os.path.exists(self._dataset_dir):
            print("Invalid dataset dir.")
            exit()

        self._anno_dir = os.path.join(self._dataset_dir, 'Anno')
        if not os.path.exists(self._anno_dir):
            print("Missing Annotation data.")
            exit()

        self._source_image_dir = os.path.join(self._dataset_dir,'img')
        if not os.path.exists(self._source_image_dir):
            print("Missing source image data.")
            exit()

        self._dataset_spilit_file = os.path.join(self._dataset_dir,'Eval','list_eval_partition.txt')
        if not os.path.exists(self._dataset_spilit_file):
            print("Missing dataset split file.")
            exit()

        # --save dir
        if len(save_dir) == 0:
            self._save_dir = os.path.join(self._dataset_dir,'Anno_tfl')
        else:
            self._save_dir = save_dir

        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)


    def dataset_split(self):
        """
        Get "train.txt val.txt test.txt"

        :return:
        """

        trainset, valset, testset = self._parse_eval_partition_file(self._dataset_spilit_file)

        # get "train.txt"
        trainset_file = open(os.path.join(self._save_dir,'train.txt'), 'w')
        for data in trainset:
            trainset_file.write('%s\n' % data[0])
        trainset_file.close()

        # get "val.txt"
        valset_file = open(os.path.join(self._save_dir, 'val.txt'), 'w')
        for data in valset:
            valset_file.write('%s\n' % data[0])
        valset_file.close()

        # get "test.txt"
        testset_file = open(os.path.join(self._save_dir, 'test.txt'), 'w')
        for data in testset:
            testset_file.write('%s\n' % data[0])
        testset_file.close()

    def genLMDB(self, dataset):
        """
        Gen LMDB format Dataset

        :param dataset: train/val/test
        :return:
        """
        # ------Dir--------
        # --LMDB saving dir
        lmdb_path = os.path.join(self._save_dir, 'lmdb', dataset)
        lmdb_abs_path = os.path.abspath(lmdb_path)

        if not os.path.exists(lmdb_abs_path):
            os.makedirs(lmdb_abs_path)

        # get lmdb object
        env = lmdb.open(lmdb_abs_path, map_size=int(1e12))
        txn = env.begin(write=True)
        data = []

        # -----Get source annotation info-----
        dataset_list_file_path = os.path.join(self._save_dir,'%s.txt' % dataset)
        if not os.path.exists(dataset_list_file_path):
            print("Missing dataset file!")
            exit()

        # get image name list
        f = open(dataset_list_file_path, 'r')
        im_name_list = f.readlines()
        f.close()

        im_name_list = [x.strip() for x in im_name_list]
        numSample = len(im_name_list)

        # get Deep_Fashion_Landmark format data
        lm_data_file = os.path.join(self._anno_dir, 'list_landmarks.txt')
        lm_dict = self._parse_lm_file(lm_data_file)

        # =====Gen LMDB=====
        # Randomize the dataset
        random_order = np.random.permutation(numSample).tolist()

        totalWriteCount = numSample
        print ('going to write %d images..' % totalWriteCount)
        writeCount = 0

        # iteration for each image
        for count in range(numSample):
            # get real index
            idx = random_order[count]

            # --Get source image
            img = cv2.imread(os.path.join(self._source_image_dir, im_name_list[idx]))
            height = img.shape[0]
            width = img.shape[1]

            # --Get image Landmark(only one landmark set per image)
            im_lm = lm_dict[im_name_list[idx]]['landmark']

            # --Get cloth type
            im_type = lm_dict[im_name_list[idx]]['cloth_type']

            # --Landmark data format change
            # [0:5]-upper [6:9]-lower [10:17]-full
            if len(im_lm)/3 > self._num_of_max_landmark:
                print("Num of Landmark is invalid!")
                exit()

            tmp_im_lm_float_list = []
            for idx in range(0, len(im_lm), 3):
                if idx + 1 > len(im_lm):
                    break

                lm_visible = im_lm[idx]
                # change from (0-visible,1-invisible,2-truncated)-->(0-invisible,1-visible,2-truncated)
                if lm_visible == 1:
                    lm_visible = 0
                elif lm_visible == 0:
                    lm_visible = 1

                lm_x = im_lm[idx + 1]
                lm_y = im_lm[idx + 2]

                tmp = [float(lm_x), float(lm_y), float(lm_visible)]
                tmp_im_lm_float_list.append(tmp)

            # upper
            if im_type == 1:
                append_head = 0
                append_tail = 12
            # lower
            elif im_type == 2:
                append_head = 6
                append_tail = 8
            # full
            else:
                append_head = 10
                append_tail = 0

            # append list
            tmp_list = []
            append_data = [-10.0, -10.0, 2.0] # TODO: Ensure this lm outside the image.
            for index in range(append_head):
                tmp_list.append(append_data)

            tmp_im_lm_float_list = tmp_list + tmp_im_lm_float_list

            for index in range(append_tail):
                tmp_im_lm_float_list.append(append_data)

            im_lm_float_list = tmp_im_lm_float_list

            # --Predict objpos from landmark data
            im_lm_arr = np.array(im_lm)
            im_lm_x_y = im_lm_arr.reshape((-1, 3))[:, 1:]

            topleft_x = np.min(im_lm_x_y[:, 0])
            topleft_y = np.min(im_lm_x_y[:, 1])
            bottomdown_x = np.max(im_lm_x_y[:, 0])
            bottomdown_y = np.max(im_lm_x_y[:, 1])

            objpos = [(topleft_x + bottomdown_x)/2, (topleft_y + bottomdown_y)/2]

            # --Padding the image if its width too small
            if width < self._image_min_width:
                img = cv2.copyMakeBorder(img, 0, 0, 0, self._image_min_width - width, cv2.BORDER_CONSTANT, value=(128, 128, 128))
                print 'saving padded image!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
                cv2.imwrite('padded_img.jpg', img)
                width = self._image_min_width
                # no modify on width, because we want to keep information

            # -----Gen Annotation Data(meta_data)-----
            # meta_data is for saving annotation info, will be concatenated with "image" data
            meta_data = np.zeros(shape=(height, width, 1), dtype=np.uint8)

            clidx = 0  # current line index

            # dataset name (string)
            for i in range(len(self._dataset_name)):
                meta_data[clidx][i] = ord(self._dataset_name[i])
            clidx = clidx + 1

            # image height, image width
            height_binary = self._float2bytes(float(height))
            for i in range(len(height_binary)):
                meta_data[clidx][i] = ord(height_binary[i])
            width_binary = self._float2bytes(float(width))
            for i in range(len(width_binary)):
                meta_data[clidx][4 + i] = ord(width_binary[i])
            clidx = clidx + 1

            # (a) isValidation(uint8), numOtherPeople (uint8), people_index (uint8), annolist_index (float), writeCount(float), totalWriteCount(float)
            meta_data[clidx][0] = float(0)  # 0 isValidation
            meta_data[clidx][1] = float(0)  # 1 numOtherPeople
            meta_data[clidx][2] = float(1)  # 2 people_index
            annolist_index_binary = self._float2bytes(float(count+1))  # index in current dataset
            for i in range(len(annolist_index_binary)):  # 3 annolist_index
                meta_data[clidx][3 + i] = ord(annolist_index_binary[i])
            count_binary = self._float2bytes(float(writeCount))  # note it's writecount instead of count! TODO: at current application count=writeCount
            for i in range(len(count_binary)):  # 4 writeCount
                meta_data[clidx][7 + i] = ord(count_binary[i])
            totalWriteCount_binary = self._float2bytes(float(totalWriteCount))
            for i in range(len(totalWriteCount_binary)):  # 5 totalWriteCount
                meta_data[clidx][11 + i] = ord(totalWriteCount_binary[i])
            nop = int(0)  # 6 numOtherPeople
            clidx = clidx + 1

            # (b) objpos_x (float), objpos_y (float)
            # TODO: object pos is predict by landmark.
            objpos_binary = self._float2bytes([float(objpos[0]), float(objpos[1])])  # objpos
            for i in range(len(objpos_binary)):
                meta_data[clidx][i] = ord(objpos_binary[i])
            clidx = clidx + 1

            # (c) scale_provided (float)
            # TODO: set "scale to 1.0 by default", which mean image do no resize.
            scale_provided_binary = self._float2bytes(float(1.0))
            for i in range(len(scale_provided_binary)):
                meta_data[clidx][i] = ord(scale_provided_binary[i])
            clidx = clidx + 1

            # (d) joint_self (3*16) or (3*22) (float) (3 line)
            joints = np.asarray(im_lm_float_list).T.tolist()  # transpose to 3*16
            print('%d %d %d'%(meta_data.shape[0],meta_data.shape[1],meta_data.shape[2]))
            print('len(joints):%d'%len(joints))
            for i in range(len(joints)):
                row_binary = self._float2bytes(joints[i])
                print('len(row_binary):%d' % len(row_binary))
                for j in range(len(row_binary)):
                    meta_data[clidx][j] = ord(row_binary[j])
                clidx = clidx + 1

            #
            img4ch = np.concatenate((img, meta_data), axis=2)
            img4ch = np.transpose(img4ch, (2, 0, 1))  # hwc->chw
            # print img4ch.shape
            datum = caffe.io.array_to_datum(img4ch, label=0)
            key = '%07d' % writeCount
            txn.put(key, datum.SerializeToString())

            # --save LMDB data every 1000 iteration
            if writeCount % 1000 == 0:
                txn.commit()
                txn = env.begin(write=True)
            print 'count: %d/ write count: %d/ randomized: %d/ all: %d' % (count, writeCount, idx, totalWriteCount)

            writeCount = writeCount + 1

        # -----finish-----
        txn.commit()
        env.close()

    # ----------------------------
    # Internal Function
    def _parse_lm_file(self, path):
        lm_dict = {}

        f = open(path)
        data = f.readlines()
        f.close()

        # remove first two line
        data = data[2:]

        # parse
        for line in data:
            data = line.strip().split(' ')
            arr_data = np.array(data)
            idx = np.where(arr_data != '')
            arr_data = arr_data[idx]

            img_name = arr_data[0].split('/')[1]
            cloth_type = int(arr_data[1])
            pose = int(arr_data[2])
            lm_str = arr_data[3:]
            lm = [int(x) for x in lm_str]

            lm_dict[img_name] = {'cloth_type':cloth_type, 'pose':pose, 'landmark':lm}

        return lm_dict

    def _parse_bb_file(self, path):
        bb_dict = {}

        f = open(path)
        data = f.readlines()
        f.close()

        # remove first two line
        data = data[2:]

        # parse
        for line in data:
            data = line.strip().split(' ')
            arr_data = np.array(data)
            idx = np.where(arr_data != '')
            arr_data = arr_data[idx]
            img_name = arr_data[0].split('/')[1]
            bb_str = arr_data[1:]
            bb = [int(x) for x in bb_str]

            bb_dict[img_name] = {'bbox':bb}

        return bb_dict

    def _parse_joint_file(self, path):
        joint_dict = {}

        f = open(path)
        data = f.readlines()
        f.close()

        # remove first two line
        data = data[2:]

        # parse
        for line in data:
            data = line.strip().split(' ')
            arr_data = np.array(data)
            idx = np.where(arr_data != '')
            arr_data = arr_data[idx]
            img_name = arr_data[0].split('/')[1]
            cloth_type = int(arr_data[1])
            pose = int(arr_data[2])

            if len(line) < 4:
                joint_dict[img_name] = {'cloth_type': cloth_type, 'pose': pose, 'joints': []}
                continue
            joint_str = arr_data[3:]
            joints = [int(x) for x in joint_str]

            joint_dict[img_name] = {'cloth_type':cloth_type, 'pose':pose, 'joints':joints}

        return joint_dict

    def _parse_eval_partition_file(self, path):
        train_list = []
        val_list = []
        test_list = []

        f = open(path)
        data = f.readlines()
        f.close()

        # remove first two line
        data = data[2:]

        # parse
        for line in data:
            data = line.strip().split(' ')
            arr_data = np.array(data)
            idx = np.where(arr_data != '')
            arr_data = arr_data[idx]
            img_name = arr_data[0].split('/')[1]
            ds = arr_data[1]

            if ds == 'train':
                train_list.append([img_name,ds])
            elif ds == 'val':
                val_list.append([img_name,ds])
            elif ds == 'test':
                test_list.append([img_name,ds])
            else:
                print("false label in list_eval_partition.txt file")
                exit()

        return train_list, val_list, test_list

    def _float2bytes(self, floats):
        if type(floats) is float:
            floats = [floats]
        return struct.pack('%sf' % len(floats), *floats)

if __name__ == "__main__":

    source_dir = '/home/tfl/workspace/dataSet/DeepFashion/Fashion_Landmark_Detection_Benchmark'
    save_dir = '/home/tfl/workspace/project/convolutional-pose-machines-release/dataset/DeepfashionLandmark'
    my_dataset = DeepFashionLandmark(dataset_dir=source_dir, save_dir=save_dir)
    my_dataset.dataset_split()
    my_dataset.genLMDB('test100')

    pass