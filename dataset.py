from __future__ import division
import os,cv2

import numpy as np
import torch.utils.data as torch_data

# 定义dataset
class G1G2Dataset(torch_data.Dataset):
    def __init__(self, mode, test_path='MDvsFA', ReadColorImage=1):
        self.mode = mode
        self.ReadColorImage = ReadColorImage
        self.test_path = test_path

        if self.mode == 'train':
            self.imageset_dir = os.path.join('./data/train/image/')
            self.imageset_gt_dir = os.path.join('./data/train/mask/')
        elif self.mode == 'test':
            if test_path == 'MDvsFA':
                self.imageset_dir = os.path.join('./data/test/MDvsFA/image/')
                self.imageset_gt_dir = os.path.join('./data/test/MDvsFA/mask/')
            elif test_path == 'SIRST':
                self.imageset_dir = os.path.join('./data/test/SIRST/image/')
                self.imageset_gt_dir = os.path.join('./data/test/SIRST/mask/')
        else:
            raise NotImplementedError

    def __len__(self):
        if self.mode == 'train':
            return 10000 
        elif self.mode == 'test':
            if self.test_path == 'MDvsFA':
                return 100 
            elif self.test_path == 'SIRST':
                return 427
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        if self.mode == 'train':
            try:
                img_dir = os.path.join(self.imageset_dir, "%06d.png"%(idx))
                gt_dir = os.path.join(self.imageset_gt_dir, "%06d.png"%(idx))
                
                real_input = cv2.imread(img_dir, -1)
                if real_input is None:
                    # 直接跳到下一个索引
                    return self.__getitem__((idx + 1) % self.__len__())
                
                bufImg = cv2.imread(gt_dir, -1)
                if bufImg is None:
                    # 直接跳到下一个索引
                    return self.__getitem__((idx + 1) % self.__len__())
                
                # 图片读取成功，进行后续处理
                real_input = np.float32(real_input)/255.0

                if self.ReadColorImage == 0:
                    input_images = real_input * 2 - 1
                    if len(input_images.shape) == 3:
                        input_images = np.mean(input_images, axis=2)
                else:
                    if len(real_input.shape) == 3:
                        input_images = real_input[:,:,2] * 2 - 1
                    else:
                        input_images = real_input * 2 - 1
                    
                input_images = np.expand_dims(input_images, axis=0)
                
                dilated_bufImg = bufImg
                output_images = np.float32(dilated_bufImg)/255.0
                output_images = np.expand_dims(output_images, axis=0)
                
                sample_info = {}
                sample_info['input_images'] = input_images
                sample_info['output_images'] = output_images

                return sample_info
                    
            except Exception as e:
                print(f"Error loading image {idx}: {str(e)}")
                # 直接跳到下一个索引
                return self.__getitem__((idx + 1) % self.__len__())
               
        elif self.mode == 'test':
            try:
                if self.test_path == 'MDvsFA':
                    img_dir = os.path.join(self.imageset_dir, "%05d.png"%(idx))
                    gt_dir = os.path.join(self.imageset_gt_dir, "%05d.png"%(idx))
                elif self.test_path == 'SIRST':
                    img_dir = os.path.join(self.imageset_dir, f"Misc_{idx + 1}.png")
                    gt_dir = os.path.join(self.imageset_gt_dir, f"Misc_{idx + 1}_pixels0.png")

                real_input = cv2.imread(img_dir, -1)
                if real_input is None:
                    return self.__getitem__((idx + 1) % self.__len__())
                
                bufImg = cv2.imread(gt_dir, -1)
                if bufImg is None:
                    return self.__getitem__((idx + 1) % self.__len__())

                real_input = np.float32(real_input)/255.0

                if self.ReadColorImage == 0:
                    input_images = real_input * 2 - 1
                    if len(input_images.shape) == 3:
                        input_images = np.mean(input_images, axis=2)
                else:
                    if len(real_input.shape) == 3:
                        input_images = real_input[:,:,2] * 2 - 1
                    else:
                        input_images = real_input * 2 - 1
                    
                input_images = np.expand_dims(input_images, axis=0)
                
                dilated_bufImg = bufImg
                output_images = np.float32(dilated_bufImg)/255.0
                output_images = np.expand_dims(output_images, axis=0)

                sample_info = {}
                sample_info['input_images'] = input_images
                sample_info['output_images'] = output_images

                return sample_info
                    
            except Exception as e:
                print(f"Error loading image {idx}: {str(e)}")
                return self.__getitem__((idx + 1) % self.__len__())
        else:
            raise NotImplementedError