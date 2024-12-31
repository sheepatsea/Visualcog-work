from __future__ import division
import os,time,cv2

import numpy as np
from torch.utils.data import DataLoader
# 用于记录日志
import logging
import torch
import torch.nn as nn
import time
import torch.optim as optim
# from model import Generator1_CAN8, Generator2_UCAN64, discriminator
from improved_model import Generator, discriminator
from dataset import G1G2Dataset

max_epoch_num = 15
max_test_num = 12000  
mini_batch_size = 4
test_interval = 100
NO_USE_NORMALIZATION = 0     
is_training = True
max_patch_num = 140000
trainImageSize = 128
isJointTrain = False
lambda1 = 100
lambda2 = 10

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
    return F1

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
    root_result_dir = os.path.join('improved_pytorch_outputs') 
    os.makedirs(root_result_dir, exist_ok=True)

    # 当前时间，日志文件的后缀
    time_suffix = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
    # 日志文件
    log_file = os.path.join(root_result_dir, 'log_train_g1g2_{}.txt'.format(time_suffix))
    logger = create_logger(log_file)

    # 定义dataset
    trainsplit = G1G2Dataset(mode='train')
    trainset = DataLoader(trainsplit, batch_size=mini_batch_size, pin_memory=True,
                              num_workers=4, shuffle=True, drop_last=True)
    testsplit = G1G2Dataset(mode='test', test_path='MDvsFA')
    testset = DataLoader(testsplit, batch_size=1, pin_memory=True,
                              num_workers=4, shuffle=False, drop_last=True)
    
    # 定义3个Model
    g1 = Generator(scale_factor=2.0)
    g1.cuda()
    g2 = Generator(scale_factor=0.5)
    g2.cuda()
    dis = discriminator(mini_batch_size=mini_batch_size)
    dis.cuda()

    # 定义3个优化器
    optimizer_g1 = optim.Adam(g1.parameters(), lr=1e-4, betas=(0.5,0.999))
    optimizer_g2 = optim.Adam(g2.parameters(), lr=1e-4, betas=(0.5,0.999))
    optimizer_d = optim.Adam(dis.parameters(), lr=1e-5, betas=(0.5,0.999))

    # 定义loss
    loss1 = nn.BCEWithLogitsLoss()

    it = 0

    best_epoch = 0
    best_idx = 0
    best_F1 = 0

    for epoch in range(0, max_epoch_num):
        # 调整学习率
        if (epoch+1) % 10 == 0:
            for p in optimizer_g1.param_groups:
                p['lr'] *= 0.2
            for q in optimizer_g2.param_groups:
                q['lr'] *= 0.2
            for r in optimizer_g2.param_groups:
                r['lr'] *= 0.2
        # 训练一个周期
        logger.info('Now we are training epoch {}!'.format(epoch+1))
        total_it_per_epoch = len(trainset)
        for bt_idx, data in enumerate(trainset):
            # 训练一个batch
            torch.cuda.empty_cache() # 释放之前占用的缓存
            it = it + 1
            logger.info('current iteration {}/{}, epoch {}/{}, total iteration: {}, g1 lr: {}, g2 lr: {}, Dis lr: {}'.format(
            bt_idx+1, total_it_per_epoch, epoch+1, max_epoch_num, it, float(optimizer_g1.param_groups[0]['lr']), 
            float(optimizer_g2.param_groups[0]['lr']), float(optimizer_d.param_groups[0]['lr'])))

            # 先训练判别器
            dis.train() 
            g1.eval()
            g2.eval()
            optimizer_g1.zero_grad()
            optimizer_g2.zero_grad()
            optimizer_d.zero_grad()

            # 输入输出放到cuda上
            input_images, output_images = data['input_images'], data['output_images']   # [B, 1, 128, 128]
            input_images = input_images.cuda(non_blocking=True).float()
            output_images = output_images.cuda(non_blocking=True).float()

            g1_out = g1(input_images) # [B, 1, 128, 128]
            g1_out = torch.clamp(g1_out, 0.0, 1.0)

            g2_out = g2(input_images) # [B, 1, 128, 128]
            g2_out = torch.clamp(g2_out, 0.0, 1.0)

            pos1 = torch.cat([input_images, 2 * output_images - 1], dim=1) # [B, 2, 128, 128]
            neg1 = torch.cat([input_images, 2 * g1_out - 1], dim=1) # [B, 2, 128, 128]  
            neg2 = torch.cat([input_images, 2 * g2_out - 1], dim=1) # [B, 2, 128, 128] 
            disc_input  = torch.cat([pos1, neg1, neg2], dim=0) # [3*B, 2, 128, 128]

            logits_real, logits_fake1, logits_fake2, Lgc = dis(disc_input) # [B, 3] [B, 3] [B, 3] [B, 1]

            const1 = torch.ones(mini_batch_size, 1).cuda(non_blocking=True).float()
            const0 = torch.zeros(mini_batch_size, 1).cuda(non_blocking=True).float()

            gen_gt  = torch.cat([const1, const0, const0], dim=1)
            gen_gt1 = torch.cat([const0, const1, const0], dim=1)
            gen_gt2 = torch.cat([const0, const0, const1], dim=1)

            ES0 = torch.mean(loss1(logits_real, gen_gt)) 
            ES1 = torch.mean(loss1(logits_fake1, gen_gt1)) 
            ES2 = torch.mean(loss1(logits_fake2, gen_gt2))

            disc_loss = ES0 + ES1 + ES2
            logger.info(" discriminator loss is {}".format(disc_loss))
            disc_loss.backward()  # 将误差反向传播
            optimizer_d.step()  # 更新参数

            # 再训练g1
            dis.eval() 
            g1.train()
            g2.eval()
            optimizer_g1.zero_grad()
            optimizer_g2.zero_grad()
            optimizer_d.zero_grad()

            g1_out = g1(input_images) # [B, 1, 128, 128]
            g1_out = torch.clamp(g1_out, 0.0, 1.0)
            MD1 = torch.mean(torch.mul(torch.pow(g1_out - output_images, 2), output_images))
            FA1 = torch.mean(torch.mul(torch.pow(g1_out - output_images, 2), 1 - output_images))
            MF_loss1 = lambda1 * MD1 + FA1 

            g2_out = g2(input_images) # [B, 1, 128, 128]
            g2_out = torch.clamp(g2_out, 0.0, 1.0)

            pos1 = torch.cat([input_images, 2 * output_images - 1], dim=1) # [B, 2, 128, 128]
            neg1 = torch.cat([input_images, 2 * g1_out - 1], dim=1) # [B, 2, 128, 128]  
            neg2 = torch.cat([input_images, 2 * g2_out - 1], dim=1) # [B, 2, 128, 128] 
            disc_input  = torch.cat([pos1, neg1, neg2], dim=0) # [3*B, 2, 128, 128]

            logits_real, logits_fake1, logits_fake2, Lgc = dis(disc_input) # [B, 3] [B, 3] [B, 3] [B, 1]

            const1 = torch.ones(mini_batch_size, 1).cuda(non_blocking=True).float()
            const0 = torch.zeros(mini_batch_size, 1).cuda(non_blocking=True).float()

            gen_gt  = torch.cat([const1, const0, const0], dim=1)
            gen_gt1 = torch.cat([const0, const1, const0], dim=1)
            gen_gt2 = torch.cat([const0, const0, const1], dim=1)


            gen_adv_loss1 = torch.mean(loss1(logits_fake1, gen_gt))
            gen_loss1  = 100*MF_loss1 + 10*gen_adv_loss1 + 1*Lgc
            logger.info(" g1 loss is {}".format(gen_loss1))

            gen_loss1.backward()  # 将误差反向传播
            optimizer_g1.step()  # 更新参数

            # 再训练g2
            dis.eval() 
            g1.eval()
            g2.train()
            optimizer_g1.zero_grad()
            optimizer_g2.zero_grad()
            optimizer_d.zero_grad()

            g1_out = g1(input_images) # [B, 1, 128, 128]
            g1_out = torch.clamp(g1_out, 0.0, 1.0)

            g2_out = g2(input_images) # [B, 1, 128, 128]
            g2_out = torch.clamp(g2_out, 0.0, 1.0)
            MD2 = torch.mean(torch.mul(torch.pow(g2_out - output_images, 2), output_images))
            FA2 = torch.mean(torch.mul(torch.pow(g2_out - output_images, 2), 1 - output_images))
            MF_loss2 = MD2 + lambda2 * FA2

            pos1 = torch.cat([input_images, 2 * output_images - 1], dim=1) # [B, 2, 128, 128]
            neg1 = torch.cat([input_images, 2 * g1_out - 1], dim=1) # [B, 2, 128, 128]  
            neg2 = torch.cat([input_images, 2 * g2_out - 1], dim=1) # [B, 2, 128, 128] 
            disc_input  = torch.cat([pos1, neg1, neg2], dim=0) # [3*B, 2, 128, 128]

            logits_real, logits_fake1, logits_fake2, Lgc = dis(disc_input) # [B, 3] [B, 3] [B, 3] [B, 1]

            const1 = torch.ones(mini_batch_size, 1).cuda(non_blocking=True).float()
            const0 = torch.zeros(mini_batch_size, 1).cuda(non_blocking=True).float()

            gen_gt  = torch.cat([const1, const0, const0], dim=1)
            gen_gt1 = torch.cat([const0, const1, const0], dim=1)
            gen_gt2 = torch.cat([const0, const0, const1], dim=1)


            gen_adv_loss2 = torch.mean(loss1(logits_fake2, gen_gt)) 
            gen_loss2  = 100*MF_loss2 + 10*gen_adv_loss2 + 1*Lgc
            logger.info(" g2 loss is {}".format(gen_loss2))

            gen_loss2.backward()  # 将误差反向传播
            optimizer_g2.step()  # 更新参数

            if (bt_idx+1) % test_interval == 0:
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
                
                for bt_idx_test, data in enumerate(testset):
                    g1.eval()
                    g2.eval()
                    dis.eval()
                    optimizer_g1.zero_grad()
                    optimizer_g2.zero_grad()
                    optimizer_d.zero_grad()

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
                    # 算g1
                    val_loss_g1 = np.mean(np.square(g1_out - output_images))
                    sum_val_loss_g1 += val_loss_g1
                    val_false_ratio_g1 = np.mean(np.maximum(0, g1_out - output_images))
                    sum_val_false_ratio_g1 += val_false_ratio_g1
                    val_detect_ratio_g1 = np.sum(g1_out * output_images)/np.maximum(np.sum(output_images),1)
                    sum_val_detect_ratio_g1 += val_detect_ratio_g1
                    val_F1_g1 = calculateF1Measure(g1_out, output_images, 0.5)
                    sum_val_F1_g1 += val_F1_g1

                    # 算g2
                    val_loss_g2 = np.mean(np.square(g2_out - output_images))
                    sum_val_loss_g2 += val_loss_g2
                    val_false_ratio_g2 = np.mean(np.maximum(0, g2_out - output_images))
                    sum_val_false_ratio_g2 += val_false_ratio_g2
                    val_detect_ratio_g2 = np.sum(g2_out * output_images)/np.maximum(np.sum(output_images),1)
                    sum_val_detect_ratio_g2 += val_detect_ratio_g2
                    val_F1_g2 = calculateF1Measure(g2_out, output_images, 0.5)
                    sum_val_F1_g2 += val_F1_g2

                    # 算g3
                    val_loss_g3 = np.mean(np.square(g3_out - output_images))
                    sum_val_loss_g3 += val_loss_g3
                    val_false_ratio_g3 = np.mean(np.maximum(0, g3_out - output_images))
                    sum_val_false_ratio_g3 += val_false_ratio_g3
                    val_detect_ratio_g3 = np.sum(g3_out * output_images)/np.maximum(np.sum(output_images),1)
                    sum_val_detect_ratio_g3 += val_detect_ratio_g3
                    val_F1_g3 = calculateF1Measure(g3_out, output_images, 0.5)
                    sum_val_F1_g3 += val_F1_g3

                    # 保存图片
                    output_image1 = np.squeeze(g1_out*255.0)#/np.maximum(output_image1.max(),0.0001))
                    output_image2 = np.squeeze(g2_out*255.0)#/np.maximum(output_image2.max(),0.0001))
                    output_image3 = np.squeeze(g3_out*255.0)#/np.maximum(output_image3.max(),0.0001))
                    #cv2.imwrite("%s/%05d_grt.png"%(task,ind),np.uint8(np.squeeze(gt_image*255.0)))
                    cv2.imwrite("improved_pytorch_outputs/results/%05d_G1.png"%(bt_idx_test),np.uint8(output_image1))
                    cv2.imwrite("improved_pytorch_outputs/results/%05d_G2.png"%(bt_idx_test),np.uint8(output_image2))
                    cv2.imwrite("improved_pytorch_outputs/results/%05d_Res.png"%(bt_idx_test),np.uint8(output_image3))                

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
                avg_val_F1_g3 = sum_val_F1_g3/len(testset)

                logger.info("================val_L2_loss is %f"% (avg_val_loss_g3))
                logger.info("================falseAlarm_rate is %f"% (avg_val_false_ratio_g3))
                logger.info("================detection_rate is %f"% (avg_val_detect_ratio_g3))
                logger.info("================F1 measure is %f"% (avg_val_F1_g3))

                
                logger.info("======================== best results ============================")
                if avg_val_F1_g3 > best_F1:
                    best_epoch = epoch
                    best_idx = bt_idx
                    best_F1 = avg_val_F1_g3

                logger.info("================Best F1 measure is %f"% (best_F1))
                logger.info("================Best Model is epoch_{}_batch_{}".format(best_epoch+1, best_idx+1))

                ############# save model
                ckpt_name1 = os.path.join(root_result_dir, 'models/g1_epoch_{}_batch_{}'.format(epoch+1, bt_idx+1))
                ckpt_name2 = os.path.join(root_result_dir, 'models/g2_epoch_{}_batch_{}'.format(epoch+1, bt_idx+1))
                ckpt_name3 = os.path.join(root_result_dir, 'models/dis_epoch_{}_batch_{}'.format(epoch+1, bt_idx+1))
                save_checkpoint(checkpoint_state(g1, optimizer_g1, epoch+1, it), filename=ckpt_name1)
                save_checkpoint(checkpoint_state(g2, optimizer_g2, epoch+1, it), filename=ckpt_name2)
                save_checkpoint(checkpoint_state(dis, optimizer_d, epoch+1, it), filename=ckpt_name3)
            




                


