import os
import numpy as np
import math
import cv2

path_epoch1 = "/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_ckt_0001/"
path_epoch2 = "/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_ckt_0002/"
path_epoch3 = "/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_ckt_0003/"
path_epoch4 = "/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_ckt_0004/"
path_epoch5 = "/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_ckt_0005/"
path_epoch6 = "/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_ckt_0006/"
path_epoch7 = "/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_ckt_0007/"
path_epoch8 = "/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_ckt_0008/"
path_epoch9 = "/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_ckt_0009/"
path_epoch10 = "/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_ckt_0010/"
path_epoch11 = "/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_ckt_0011/"
path_epoch12 = "/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_ckt_0012/"
path_epoch17 = "/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_ckt_0017/"

path_noshare_epoch1="/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_noshare_ckt_0001/"
path_noshare_epoch2="/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_noshare_ckt_0002/"
path_noshare_epoch3="/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_noshare_ckt_0003/"
path_noshare_epoch4="/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_noshare_ckt_0004/"
path_noshare_epoch5="/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_noshare_ckt_0005/"
path_noshare_epoch6="/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_noshare_ckt_0006/"
path_noshare_epoch7="/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_noshare_ckt_0007/"
path_noshare_epoch8="/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_noshare_ckt_0008/"
path_noshare_epoch9="/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_noshare_ckt_0009/"
path_noshare_epoch10="/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_noshare_ckt_0010/"
path_noshare_epoch11="/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_noshare_ckt_0011/"
path_noshare_epoch12="/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_noshare_ckt_0012/"
path_noshare_epoch17="/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_noshare_ckt_0017/"

path_ensem_epoch1="/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_noshare_ckt_0001/"
path_ensem_epoch2="/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_noshare_ckt_0002/"
path_ensem_epoch3="/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_noshare_ckt_0003/"
path_ensem_epoch4="/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_noshare_ckt_0004/"
path_ensem_epoch5="/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_noshare_ckt_0005/"
path_ensem_epoch6="/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_noshare_ckt_0006/"
path_ensem_epoch7="/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_noshare_ckt_0007/"
path_ensem_epoch8="/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_noshare_ckt_0008/"
path_ensem_epoch9="/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_noshare_ckt_0009/"
path_ensem_epoch10="/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_noshare_ckt_0010/"
path_ensem_epoch11="/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_noshare_ckt_0011/"
path_ensem_epoch12="/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_noshare_ckt_0012/"
path_ensem_epoch17="/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/baseline_result_ep_EnsemableTeacher_corr_OnlineDistill_confidence_noshare_ckt_0017/"

path_gt = "/home/xteam/yh/code/test_gt_image/"
path_input = "/opt/dataset/HED-BSDS/test/"
path_save= "/home/xteam/yh/code/0_816_version_EnsemableTeacher_corr_OnlineDistill_confidence/visulization_results_confidence/"

filenames = os.listdir(path_epoch1)
for filename in filenames:
    img1 = cv2.imread(path_epoch1 + filename).astype(np.float64)
    img2 = cv2.imread(path_epoch2 + filename).astype(np.float64)
    img3 = cv2.imread(path_epoch3 + filename).astype(np.float64)
    img4 = cv2.imread(path_epoch4 + filename).astype(np.float64)
    img5 = cv2.imread(path_epoch5 + filename).astype(np.float64)
    img6 = cv2.imread(path_epoch6 + filename).astype(np.float64)
    img7 = cv2.imread(path_epoch7 + filename).astype(np.float64)
    img8 = cv2.imread(path_epoch8 + filename).astype(np.float64)
    img9 = cv2.imread(path_epoch9 + filename).astype(np.float64)
    img10 = cv2.imread(path_epoch10 + filename).astype(np.float64)
    img11 = cv2.imread(path_epoch11 + filename).astype(np.float64)
    img12 = cv2.imread(path_epoch12 + filename).astype(np.float64)
    img17 = cv2.imread(path_epoch17 + filename).astype(np.float64)

    img_noshare1 = cv2.imread(path_noshare_epoch1 + filename).astype(np.float64)
    img_noshare2 = cv2.imread(path_noshare_epoch2 + filename).astype(np.float64)
    img_noshare3 = cv2.imread(path_noshare_epoch3 + filename).astype(np.float64)
    img_noshare4 = cv2.imread(path_noshare_epoch4 + filename).astype(np.float64)
    img_noshare5 = cv2.imread(path_noshare_epoch5 + filename).astype(np.float64)
    img_noshare6 = cv2.imread(path_noshare_epoch6 + filename).astype(np.float64)
    img_noshare7 = cv2.imread(path_noshare_epoch7 + filename).astype(np.float64)
    img_noshare8 = cv2.imread(path_noshare_epoch8 + filename).astype(np.float64)
    img_noshare9 = cv2.imread(path_noshare_epoch9 + filename).astype(np.float64)
    img_noshare10 = cv2.imread(path_noshare_epoch10 + filename).astype(np.float64)
    img_noshare11 = cv2.imread(path_noshare_epoch11 + filename).astype(np.float64)
    img_noshare12 = cv2.imread(path_noshare_epoch12 + filename).astype(np.float64)
    img_noshare17 = cv2.imread(path_noshare_epoch17 + filename).astype(np.float64)

    img_ensem1 = cv2.imread(path_ensem_epoch1 + filename).astype(np.float64)
    img_ensem2 = cv2.imread(path_ensem_epoch2 + filename).astype(np.float64)
    img_ensem3 = cv2.imread(path_ensem_epoch3 + filename).astype(np.float64)
    img_ensem4 = cv2.imread(path_ensem_epoch4 + filename).astype(np.float64)
    img_ensem5 = cv2.imread(path_ensem_epoch5 + filename).astype(np.float64)
    img_ensem6 = cv2.imread(path_ensem_epoch6 + filename).astype(np.float64)
    img_ensem7 = cv2.imread(path_ensem_epoch7 + filename).astype(np.float64)
    img_ensem8 = cv2.imread(path_ensem_epoch8 + filename).astype(np.float64)
    img_ensem9 = cv2.imread(path_ensem_epoch9 + filename).astype(np.float64)
    img_ensem10 = cv2.imread(path_ensem_epoch10 + filename).astype(np.float64)
    img_ensem11 = cv2.imread(path_ensem_epoch11 + filename).astype(np.float64)
    img_ensem12 = cv2.imread(path_ensem_epoch12 + filename).astype(np.float64)
    img_ensem17 = cv2.imread(path_ensem_epoch17 + filename).astype(np.float64)

    filename_jpg = filename.replace('.png', '.jpg')
    gt = cv2.imread(path_gt + filename_jpg).astype(np.float64)
    img = cv2.imread(path_input + filename_jpg).astype(np.float64)

    cat1 = np.concatenate((img, img1, img_noshare1, img_ensem1, gt), axis=1)
    cat2 = np.concatenate((img, img2, img_noshare2, img_ensem2, gt), axis=1)
    cat3 = np.concatenate((img, img3, img_noshare3, img_ensem3, gt), axis=1)
    cat4 = np.concatenate((img, img4, img_noshare4, img_ensem4, gt), axis=1)
    cat5 = np.concatenate((img, img5, img_noshare5, img_ensem5, gt), axis=1)
    cat6 = np.concatenate((img, img6, img_noshare6, img_ensem6, gt), axis=1)
    cat7 = np.concatenate((img, img7, img_noshare7, img_ensem7, gt), axis=1)
    cat8 = np.concatenate((img, img8, img_noshare8, img_ensem8, gt), axis=1)
    cat9 = np.concatenate((img, img9, img_noshare9, img_ensem9, gt), axis=1)
    cat10 = np.concatenate((img, img10, img_noshare10, img_ensem10, gt), axis=1)
    cat11 = np.concatenate((img, img11, img_noshare11, img_ensem11, gt), axis=1)
    cat12 = np.concatenate((img, img12, img_noshare12, img_ensem12, gt), axis=1)
    cat17 = np.concatenate((img, img17, img_noshare17, img_ensem17, gt), axis=1)

    cat = np.concatenate((cat1, cat2, cat3, cat4, cat5, cat6, cat7, cat8, cat9, cat10, cat11, cat12, cat17), axis=0)
    cv2.imwrite(path_save+filename, cat) 




