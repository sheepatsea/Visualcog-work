from __future__ import division
import os,time,cv2

import numpy as np
from torch.utils.data import DataLoader
# 用于记录日志
import logging
import torch
import time
from model import Generator1_CAN8, Generator2_UCAN64, discriminator
from dataset import G1G2Dataset

max_epoch_num = 15
max_test_num = 12000  
mini_batch_size = 10
NO_USE_NORMALIZATION = 0     
is_training = True
max_patch_num = 140000
trainImageSize = 128
isJointTrain = False
lambda1 = 100
lambda2 = 10
best_path = 'epoch_3_batch_90.pth'

##################################################################################
def create_logger(log_file):
    # 定义好logger
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)
    console = logging.StreamHandler() # 日志输出到流
    console.setLevel(logging.INFO) # 日志等级
    console.setFormatter(logging.Formatter(log_format)) # 设置日志格式
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)

def calculateF1Measure(output_image,gt_image,thre):
    output_image = np.squeeze(output_image)
    gt_image = np.squeeze(gt_image)
    out_bin = output_image>thre
    gt_bin = gt_image>thre
    recall = np.sum(gt_bin*out_bin)/np.maximum(1,np.sum(gt_bin))
    prec   = np.sum(gt_bin*out_bin)/np.maximum(1,np.sum(out_bin))
    F1 = 2*recall*prec/np.maximum(0.001,recall+prec)
    return F1, prec, recall

def save_checkpoint(state, filename='checkpoint'):
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)

def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state}




if __name__ == '__main__':
    # 保存输出的总路径
    root_result_dir = os.path.join('pytorch_outputs') 
    os.makedirs(root_result_dir, exist_ok=True)

    # 当前时间，日志文件的后缀
    time_suffix = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
    # 日志文件
    log_file = os.path.join(root_result_dir, 'log_train_g1g2_{}.txt'.format(time_suffix))
    logger = create_logger(log_file)

    # 定义dataset
    # trainsplit = G1G2Dataset(mode='train')
    # trainset = DataLoader(trainsplit, batch_size=mini_batch_size, pin_memory=True,
    #                           num_workers=4, shuffle=True, drop_last=True)
    testsplit = G1G2Dataset(mode='test', test_path='SIRST')
    testset = DataLoader(testsplit, batch_size=1, pin_memory=True,
                              num_workers=4, shuffle=False, drop_last=True)
    
    # 定义3个Model
    g1 = Generator1_CAN8()
    g1.cuda()
    g2 = Generator2_UCAN64()
    g2.cuda()
    dis = discriminator(mini_batch_size=10)
    dis.cuda()
    checkpoint_g1 = torch.load('./pytorch_outputs/models/'+'g1_'+best_path) 
    g1.load_state_dict(checkpoint_g1['model_state']) 
    checkpoint_g2 = torch.load('./pytorch_outputs/models/'+'g2_'+best_path) 
    g2.load_state_dict(checkpoint_g2['model_state']) 
    checkpoint_dis = torch.load('./pytorch_outputs/models/'+'dis_'+best_path) 
    dis.load_state_dict(checkpoint_dis['model_state']) 

    g1.eval()
    g2.eval()
    dis.eval()  


    # 在测试集上测试
    sum_val_loss_g1 = 0
    sum_val_false_ratio_g1 = 0 
    sum_val_detect_ratio_g1 = 0
    sumRealTarN_g1 = 0
    sumDetTarN_g1 = 0
    sum_val_F1_g1 = 0
    g1_time = 0

    sum_val_loss_g2 = 0
    sum_val_false_ratio_g2 = 0 
    sum_val_detect_ratio_g2 = 0
    sumRealTarN_g2 = 0
    sumDetTarN_g2 = 0
    sum_val_F1_g2 = 0
    g2_time = 0

    sum_val_loss_g3 = 0
    sum_val_false_ratio_g3 = 0 
    sum_val_detect_ratio_g3 = 0
    sumRealTarN_g3 = 0
    sumDetTarN_g3 = 0
    sum_val_F1_g3 = 0
    sum_prec_g3 = 0
    sum_recall_g3 = 0

    num = 0

    for bt_idx_test, data in enumerate(testset):

        # 将输入输出放到cuda上
        input_images, output_images = data['input_images'], data['output_images']   # [B, 1, 128, 128]
        input_images = input_images.cuda(non_blocking=True).float()
        output_images = output_images.cuda(non_blocking=True).float()

        stime = time.time()
        g1_out = g1(input_images) # [B, 1, 128, 128]
        etime = time.time()
        g1_time += etime - stime
        logger.info('testing {}, g1 time is {}'.format(bt_idx_test, etime-stime))
        g1_out = torch.clamp(g1_out, 0.0, 1.0)

        stime = time.time()
        g2_out = g2(input_images) # [B, 1, 128, 128]
        etime = time.time()
        g2_time += etime - stime
        logger.info('testing {}, g2 time is {}'.format(bt_idx_test, etime-stime))
        g2_out = torch.clamp(g2_out, 0.0, 1.0)

        g3_out = (g1_out + g2_out) / 2 # 取均值的方式进行融合

        output_images = output_images.cpu().numpy()
        g1_out = g1_out.detach().cpu().numpy()
        g2_out = g2_out.detach().cpu().numpy()
        g3_out = g3_out.detach().cpu().numpy()
        if g1_out.shape != output_images.shape:
            num += 1
            continue
        # 算g1
        val_loss_g1 = np.mean(np.square(g1_out - output_images))
        sum_val_loss_g1 += val_loss_g1
        val_false_ratio_g1 = np.mean(np.maximum(0, g1_out - output_images))
        sum_val_false_ratio_g1 += val_false_ratio_g1
        val_detect_ratio_g1 = np.sum(g1_out * output_images)/np.maximum(np.sum(output_images),1)
        sum_val_detect_ratio_g1 += val_detect_ratio_g1
        val_F1_g1,_,_ = calculateF1Measure(g1_out, output_images, 0.5)
        sum_val_F1_g1 += val_F1_g1

        # 算g2
        val_loss_g2 = np.mean(np.square(g2_out - output_images))
        sum_val_loss_g2 += val_loss_g2
        val_false_ratio_g2 = np.mean(np.maximum(0, g2_out - output_images))
        sum_val_false_ratio_g2 += val_false_ratio_g2
        val_detect_ratio_g2 = np.sum(g2_out * output_images)/np.maximum(np.sum(output_images),1)
        sum_val_detect_ratio_g2 += val_detect_ratio_g2
        val_F1_g2,_,_ = calculateF1Measure(g2_out, output_images, 0.5)
        sum_val_F1_g2 += val_F1_g2

        # 算g3
        val_loss_g3 = np.mean(np.square(g3_out - output_images))
        sum_val_loss_g3 += val_loss_g3
        val_false_ratio_g3 = np.mean(np.maximum(0, g3_out - output_images))
        sum_val_false_ratio_g3 += val_false_ratio_g3
        val_detect_ratio_g3 = np.sum(g3_out * output_images)/np.maximum(np.sum(output_images),1)
        sum_val_detect_ratio_g3 += val_detect_ratio_g3
        val_F1_g3, prec_g3, recall_g3 = calculateF1Measure(g3_out, output_images, 0.5)
        sum_val_F1_g3 += val_F1_g3
        sum_prec_g3 += prec_g3
        sum_recall_g3 += recall_g3

        # 保存图片
        output_image1 = np.squeeze(g1_out*255.0)#/np.maximum(output_image1.max(),0.0001))
        output_image2 = np.squeeze(g2_out*255.0)#/np.maximum(output_image2.max(),0.0001))
        output_image3 = np.squeeze(g3_out*255.0)#/np.maximum(output_image3.max(),0.0001))
        #cv2.imwrite("%s/%05d_grt.png"%(task,ind),np.uint8(np.squeeze(gt_image*255.0)))
        cv2.imwrite("pytorch_outputs/results/SIRST/%05d_G1.png"%(bt_idx_test),np.uint8(output_image1))
        cv2.imwrite("pytorch_outputs/results/SIRST/%05d_G2.png"%(bt_idx_test),np.uint8(output_image2))
        cv2.imwrite("pytorch_outputs/results/SIRST/%05d_Res.png"%(bt_idx_test),np.uint8(output_image3))                

    logger.info("======================== g1 results ============================")
    avg_val_loss_g1 = sum_val_loss_g1/len(testset)
    avg_val_false_ratio_g1  = sum_val_false_ratio_g1/len(testset)
    avg_val_detect_ratio_g1 = sum_val_detect_ratio_g1/len(testset)
    avg_val_F1_g1 = sum_val_F1_g1/len(testset)

    logger.info("================val_L2_loss is %f"% (avg_val_loss_g1))
    logger.info("================falseAlarm_rate is %f"% (avg_val_false_ratio_g1))
    logger.info("================detection_rate is %f"% (avg_val_detect_ratio_g1))
    logger.info("================F1 measure is %f"% (avg_val_F1_g1))
    logger.info("g1 time is {}".format(g1_time))

    logger.info("======================== g2 results ============================")
    avg_val_loss_g2 = sum_val_loss_g2/len(testset)
    avg_val_false_ratio_g2  = sum_val_false_ratio_g2/len(testset)
    avg_val_detect_ratio_g2 = sum_val_detect_ratio_g2/len(testset)
    avg_val_F1_g2 = sum_val_F1_g2/len(testset)

    logger.info("================val_L2_loss is %f"% (avg_val_loss_g2))
    logger.info("================falseAlarm_rate is %f"% (avg_val_false_ratio_g2))
    logger.info("================detection_rate is %f"% (avg_val_detect_ratio_g2))
    logger.info("================F1 measure is %f"% (avg_val_F1_g2))
    logger.info("g2 time is {}".format(g2_time))

    logger.info("======================== g3 results ============================")
    avg_val_loss_g3 = sum_val_loss_g3/len(testset)
    avg_val_false_ratio_g3  = sum_val_false_ratio_g3/len(testset)
    avg_val_detect_ratio_g3 = sum_val_detect_ratio_g3/len(testset)
    avg_val_F1_g3 = sum_val_F1_g3/(len(testset)-num)
    avg_prec_g3 = sum_prec_g3 / (len(testset)-num)
    avg_recall_g3 = sum_recall_g3 / (len(testset)-num)

    logger.info("================val_L2_loss is %f"% (avg_val_loss_g3))
    logger.info("================falseAlarm_rate is %f"% (avg_val_false_ratio_g3))
    logger.info("================detection_rate is %f"% (avg_val_detect_ratio_g3))
    logger.info("================F1 measure is %f"% (avg_val_F1_g3))
    logger.info("================precision is %f"% (avg_prec_g3))
    logger.info("================recall is %f"% (avg_recall_g3))
    

    
