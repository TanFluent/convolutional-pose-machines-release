import os
import pdb
import numpy as np
import cv2
import matplotlib.pyplot as plt


class DeepFashionLandmark:
    """
    Handle DeepFashionLandmark Dataset
    """
    root_dir = ''

    def __init__(self,root_dir):
        self.root_dir = root_dir

        self.cloth_type = ['None', 'upper-body', 'lower-body', 'full-body']
        self.pose_type = ['None', 'normal pose', 'medium pose', 'large pose', 'medium zoom-in', 'large zoom-in']
        self.visibility_type = ['visible', 'invisible', 'truncated']

        # TODO: TMD!!!!! Real orders is opposite with the description of "README.txt"
        # self.upper_body_lm = ["left collar", "right collar", "left sleeve", "right sleeve", "left hem", "right hem"]
        # self.lower_body_lm = ["left waistline", "right waistline", "left hem", "right hem"]
        # self.full_body_lm = ["left collar", "right collar", "left sleeve", "right sleeve", "left waistline",
        #                     "right waistline", "left hem", "right hem"]
        self.upper_body_lm = ["right collar", "left collar", "right sleeve", "left sleeve", "right hem", "left hem"]
        self.lower_body_lm = ["right waistline", "left waistline", "right hem", "left hem"]
        self.full_body_lm = ["right collar", "left collar", "right sleeve", "left sleeve", "right waistline",
                             "left waistline", "right hem", "left hem"]

    def show_DeepFashionLM(self):

        # Annotation
        anno_lm_file = os.path.join(self.root_dir, 'Anno', 'list_landmarks.txt')
        anno_bb_file = os.path.join(self.root_dir, 'Anno', 'list_bbox.txt')
        anno_joints_file = os.path.join(self.root_dir, 'Anno', 'list_joints.txt')

        # dataset split
        eval_partition_file = os.path.join(self.root_dir, 'Eval', 'list_eval_partition.txt')

        # source images
        source_images_dir = os.path.join(self.root_dir, 'img')

        # -parse landmark file
        lm_dict = self._parse_lm_file(anno_lm_file)

        # -parse bbox file
        bb_dict = self._parse_bb_file(anno_bb_file)

        # -parse joints file
        joint_dict = self._parse_joint_file(anno_joints_file)

        # -parse eval partition file
        train_set, val_set, test_set = self._parse_eval_partition_file(eval_partition_file)

        all_dataset = train_set + val_set + test_set

        # -show first image of the dataset
        image_idx = 0
        im_name = all_dataset[image_idx][0]
        ds_name = all_dataset[image_idx][1]
        im_path = os.path.join(source_images_dir, im_name)
        im = cv2.imread(im_path)

        im_lm = lm_dict[im_name]
        im_bb = bb_dict[im_name]
        im_joints = joint_dict[im_name]

        im_final = self._plot_anno_info(im, im_lm, im_bb, im_joints)

        win_name = '%s-%s' % (ds_name, im_name)
        cv2.imshow(win_name, im_final)
        cv2.moveWindow(win_name, 10, 10)

        while True:
            k = cv2.waitKey(0)
            if k == ord('q'):
                cv2.destroyAllWindows()
                break
            elif k == ord('n'):
                cv2.destroyWindow(win_name)
                image_idx = image_idx+1
                if image_idx > len(all_dataset):
                    image_idx = len(all_dataset) - 1
                    print("#--End of the dataset.")
            elif k == ord('p'):
                cv2.destroyWindow(win_name)
                image_idx = image_idx - 1
                if image_idx < 0:
                    image_idx = 0
                    print("#--Begin of the dataset.")

            im_name = all_dataset[image_idx][0]
            ds_name = all_dataset[image_idx][1]
            im_path = os.path.join(source_images_dir, im_name)
            im = cv2.imread(im_path)

            im_lm = lm_dict[im_name]
            im_bb = bb_dict[im_name]
            im_joints = joint_dict[im_name]

            im_final = self._plot_anno_info(im, im_lm, im_bb, im_joints)

            win_name = '%s-%s' % (ds_name, im_name)
            cv2.imshow(win_name, im_final)
            cv2.moveWindow(win_name, 10, 10)

        #pdb.set_trace()
        print('finish')

    def _plot_anno_info(self, im, lm_info, bb_info, joints_info):
        h,w,c = im.shape

        bb_color = (50, 255, 255)
        lm_color_visible = (255, 0, 0)
        lm_color_invisible = (255, 255, 0)
        joints_color = (0, 255, 0)

        # plot bb
        bb = bb_info['bbox']
        cv2.rectangle(im, (bb[0], bb[1]), (bb[2], bb[3]), bb_color, 2)

        # plot lm
        lm = lm_info['landmark']

        # cloth type
        if len(lm)/3 == 6:
            cloth_type_idx = 1
        elif len(lm)/3 == 4:
            cloth_type_idx = 2
        elif len(lm)/3 == 8:
            cloth_type_idx = 3
        else:
            print("Invalid Landmark number! Please check your data.")
            exit()

        for idx in range(0, len(lm), 3):
            if idx+1 > len(lm):
                break

            lm_visible = lm[idx + 0]  # 0 : visible; 1 : invisible
            lm_x = lm[idx + 1]
            lm_y = lm[idx + 2]

            lm_color = lm_color_visible
            if self.visibility_type[lm_visible] == "invisible":
                lm_color = lm_color_invisible
            elif self.visibility_type[lm_visible] == "truncated":
                print("landmark #%d is not in image" % (idx/3))

            cv2.circle(im, (lm_x, lm_y), 10, color=lm_color)
            cv2.putText(im, '%d'%(idx/3), (lm_x, lm_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=lm_color,
                    thickness=2)

        # plot joints
        joints = joints_info['joints']

        for idx in range(0, len(joints), 3):
            if idx + 1 > len(joints):
                    print("#--No joints data...")
                    break
            joints_x = joints[idx + 1]
            joints_y = joints[idx + 2]
            cv2.circle(im, (joints_x, joints_y), 10, joints_color)

        # plot text info
        pose_idx = lm_info['pose']
        cloth_idx = lm_info['cloth_type']

        pose_type = self.pose_type[pose_idx]
        cloth_type = self.cloth_type[cloth_idx]

        text_info = '#' + pose_type
        cv2.putText(im, text_info, (1, 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 200, 255),
                    thickness=2)
        text_info = '#' + cloth_type
        cv2.putText(im, text_info, (1, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 200, 255),
                    thickness=2)
        return im

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

if __name__ == "__main__":

    Dataset_root = '/home/tfl/workspace/dataSet/DeepFashion/Fashion_Landmark_Detection_Benchmark/'

    dataset = 'DeepFashion_LM'

    if dataset=='DeepFashion_LM':
        my_dataset = DeepFashionLandmark(Dataset_root)
        my_dataset.show_DeepFashionLM()


    pass