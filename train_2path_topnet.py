import os
import torch
import topnet
import data_utils
import numpy as np 
from datetime import datetime
from torch.autograd import Variable
from loss import emdModule as earth_mover_distance 
from data_utils import rotate_2pc, jitter_2pc, scale_2pc


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# load_models_dir = 'Nov26_single_topnet_models'
load_models_dir = 'May12_single_topnet_models_noise_10_20_30_40'
# save_model_dir = 'Nov21_topnet_2path_models'
save_model_dir = 'May15_topnet_2path_models_noise_10_20_30_40'
train_noisy_data = False # train noisy data
learning_rate = 0.001
bs = 50
epoches = 50
partial_point_cloud_number = 512
coarse_num = 2048


#1,3,5,6,8,9
for t in [1,3,5,6,8,9]:
    for split_num in range(5):
        train_split=None
        valid_split=None
        if split_num==0:
            train_split=[1,2,3,4]
            valid_split=[5]
        if split_num==1:
            train_split=[1,2,3,5]
            valid_split=[4]
        if split_num==2:
            train_split=[1,2,4,5]
            valid_split=[3]
        if split_num==3:
            train_split=[1,3,4,5]
            valid_split=[2]
        if split_num==4:
            train_split=[2,3,4,5]
            valid_split=[1]

        dir_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(dir_name)
        print("type%d"%t)
        print("fold%d"%valid_split[0])

        topnet_path1 = topnet.TopNet_path1(partial_point_cloud_number, coarse_num)
        topnet_path2 = topnet.TopNet_path2(partial_point_cloud_number, coarse_num)
        topnet_path1.load_state_dict(torch.load('./data/%s/type%d/fold%d/best_single_topnet_path1_params.pth'%(load_models_dir,t,valid_split[0])))
        topnet_path2.load_state_dict(torch.load('./data/%s/type%d/fold%d/best_single_topnet_path2_params.pth'%(load_models_dir,t,valid_split[0])))
        topnet_path1.cuda()
        topnet_path2.cuda()

        if not train_noisy_data:
            dset_train = data_utils.co_xyzDataSet_2partial_nor(dataset_path='./data/2partial_normalized_data', 
                                                type_num=t,
                                                splits=train_split)
            dataloader_train = torch.utils.data.DataLoader(dset_train, 
                                                        batch_size=bs,
                                                        shuffle=True,
                                                        num_workers=8)
            dset_valid = data_utils.co_xyzDataSet_2partial_nor(dataset_path='./data/2partial_normalized_data', 
                                                type_num=t,
                                                splits=valid_split)
            dataloader_valid = torch.utils.data.DataLoader(dset_valid, 
                                                        batch_size=bs,
                                                        shuffle=False,
                                                        num_workers=8)
        else:
            dset_train = data_utils.co_xyzDataSet_2partial_nor(dataset_path='./data/2partial_normalized_noisy_data', 
                                                type_num=t,
                                                splits=train_split)
            dataloader_train = torch.utils.data.DataLoader(dset_train, 
                                                        batch_size=bs,
                                                        shuffle=True,
                                                        num_workers=8)
            dset_valid = data_utils.co_xyzDataSet_2partial_nor(dataset_path='./data/2partial_normalized_noisy_data', 
                                                type_num=t,
                                                splits=valid_split)
            dataloader_valid = torch.utils.data.DataLoader(dset_valid, 
                                                        batch_size=bs,
                                                        shuffle=False,
                                                        num_workers=8)
        print(len(dset_train))
        print(len(dset_valid))

        EMD = earth_mover_distance()

        optimizer_path1 = torch.optim.Adam(topnet_path1.parameters(), lr=learning_rate)
        optimizer_path2 = torch.optim.Adam(topnet_path2.parameters(), lr=learning_rate)

        min_loss_path1 = 1e7
        min_loss_path2 = 1e7
        best_epoch_path1 = -1
        best_epoch_path2 = -1
        for i in range(epoches):
            ########################################################################################
            # valid 2 paths
            ########################################################################################
            total_valid_loss_path1 = []
            total_valid_loss_path2 = []
            valid_re_loss = []
            for AB_partial, gt_coarse in dataloader_valid:
                AB_partial = Variable(AB_partial).permute(0,2,1)
                gt_coarse = Variable(gt_coarse).permute(0,2,1)
                AB_partial, gt_coarse = rotate_2pc(AB_partial, gt_coarse)
                AB_partial = Variable(AB_partial).cuda().permute(0,2,1)
                gt_coarse = Variable(gt_coarse).cuda()

                topnet_path1.eval()
                topnet_path2.eval()
                A_1, B_2 = topnet_path1(AB_partial)
                A_1 = A_1.permute(0,2,1)
                B_2 = B_2.permute(0,2,1)
                B_1, A_2 = topnet_path2(AB_partial)
                B_1 = B_1.permute(0,2,1)
                A_2 = A_2.permute(0,2,1)

                re = torch.mean(EMD(A_1, A_2, 0.005, 50)[0])+torch.mean(EMD(B_1, B_2, 0.005, 50)[0])
                loss_A_1 = torch.mean(EMD(A_1, gt_coarse[:,0:int(coarse_num/2),:], 0.005, 50)[0])
                loss_B_2 = torch.mean(EMD(B_2, gt_coarse[:,int(coarse_num/2):,:], 0.005, 50)[0])
                loss_B_1 = torch.mean(EMD(B_1, gt_coarse[:,int(coarse_num/2):,:], 0.005, 50)[0])
                loss_A_2 = torch.mean(EMD(A_2, gt_coarse[:,0:int(coarse_num/2),:], 0.005, 50)[0])

                total_valid_loss_path1.append(np.sqrt(loss_A_1.item())*100+np.sqrt(loss_B_2.item())*100)
                total_valid_loss_path2.append(np.sqrt(loss_B_1.item())*100+np.sqrt(loss_A_2.item())*100)
                valid_re_loss.append(np.sqrt(re.item())*100)

            total_valid_loss_path1 = sum(total_valid_loss_path1)/len(total_valid_loss_path1)
            total_valid_loss_path2 = sum(total_valid_loss_path2)/len(total_valid_loss_path2)
            valid_re_loss = sum(valid_re_loss)/len(valid_re_loss)
            print("------------------------------------------------------------------------")
            print("valid results")
            print("[%d/%d] path1 Loss: %.2f path2 Loss: %.2f re loss: %.2f" %\
                (i, epoches, total_valid_loss_path1,total_valid_loss_path2,valid_re_loss))
            
            dir_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(dir_name)

            min_loss_path1 = min(total_valid_loss_path1, min_loss_path1)
            min_loss_path2 = min(total_valid_loss_path2, min_loss_path2)
            
            if os.path.exists(os.path.join('./saved_models',save_model_dir))==False:
                os.mkdir(os.path.join('./saved_models',save_model_dir))
            if os.path.exists(os.path.join('./saved_models',save_model_dir,'type%d'%t))==False:
                os.mkdir(os.path.join('./saved_models',save_model_dir,'type%d'%t))
            if os.path.exists(os.path.join('./saved_models',save_model_dir,'type%d'%t,'fold%d'%valid_split[0]))==False:
                os.mkdir(os.path.join('./saved_models',save_model_dir,'type%d'%t,'fold%d'%valid_split[0]))
            
            if total_valid_loss_path1==min_loss_path1:
                best_epoch_path1 = i
                torch.save(topnet_path1.state_dict(),'./saved_models/%s/type%d/fold%d/best_topnet_path1_params.pth'%(save_model_dir, t, valid_split[0]))
            if total_valid_loss_path2==min_loss_path2:
                best_epoch_path2 = i
                torch.save(topnet_path2.state_dict(),'./saved_models/%s/type%d/fold%d/best_topnet_path2_params.pth'%(save_model_dir, t, valid_split[0]))
            
            print("best epoch path1: %d min loss: %.2f"%(best_epoch_path1, min_loss_path1))
            print("best epoch path2: %d min loss: %.2f"%(best_epoch_path2, min_loss_path2))
            print("------------------------------------------------------------------------")
            ########################################################################################
            # train 2 paths
            ########################################################################################
            r_loss_1_path1 = []
            r_loss_2_path1 = []
            r_loss_1_path2 = []
            r_loss_2_path2 = []
            train_re_loss = []
            for AB_partial, gt_coarse in dataloader_train:
                AB_partial = Variable(AB_partial).permute(0,2,1)
                gt_coarse = Variable(gt_coarse).permute(0,2,1)
                AB_partial, gt_coarse = rotate_2pc(AB_partial, gt_coarse)
                AB_partial, gt_coarse = jitter_2pc(AB_partial, gt_coarse)
                AB_partial, gt_coarse = scale_2pc(AB_partial, gt_coarse)
                AB_partial = Variable(AB_partial).cuda().permute(0,2,1)
                gt_coarse = Variable(gt_coarse).cuda()
                optimizer_path1.zero_grad()
                optimizer_path2.zero_grad()

                topnet_path1.train()
                topnet_path2.train()
                A_1, B_2 = topnet_path1(AB_partial)
                A_1 = A_1.permute(0,2,1)
                B_2 = B_2.permute(0,2,1)
                B_1, A_2 = topnet_path2(AB_partial)
                B_1 = B_1.permute(0,2,1)
                A_2 = A_2.permute(0,2,1)

                re = torch.mean(EMD(A_1, A_2, 0.005, 50)[0])+torch.mean(EMD(B_1, B_2, 0.005, 50)[0])
                loss_A_1 = torch.mean(EMD(A_1, gt_coarse[:,0:int(coarse_num/2),:], 0.005, 50)[0])
                loss_B_2 = torch.mean(EMD(B_2, gt_coarse[:,int(coarse_num/2):,:], 0.005, 50)[0])
                loss_B_1 = torch.mean(EMD(B_1, gt_coarse[:,int(coarse_num/2):,:], 0.005, 50)[0])
                loss_A_2 = torch.mean(EMD(A_2, gt_coarse[:,0:int(coarse_num/2),:], 0.005, 50)[0])
                loss_path1 = loss_A_1 + loss_B_2
                loss_path2 = loss_A_2 + loss_B_1

                total_loss = 0.00001*(loss_path1 + loss_path2) + re # our weight
                # total_loss = re
                total_loss.backward()
                optimizer_path1.step()
                optimizer_path2.step()

                r_loss_1_path1.append(np.sqrt(loss_A_1.item())*100)
                r_loss_2_path1.append(np.sqrt(loss_B_2.item())*100)
                r_loss_1_path2.append(np.sqrt(loss_B_1.item())*100)
                r_loss_2_path2.append(np.sqrt(loss_A_2.item())*100)
                train_re_loss.append(np.sqrt(re.item())*100)

            r_loss_1_path1 = sum(r_loss_1_path1)/len(r_loss_1_path1)
            r_loss_2_path1 = sum(r_loss_2_path1)/len(r_loss_2_path1)
            r_loss_1_path2 = sum(r_loss_1_path2)/len(r_loss_1_path2)
            r_loss_2_path2 = sum(r_loss_2_path2)/len(r_loss_2_path2)
            train_re_loss = sum(train_re_loss)/len(train_re_loss)

            print("[%d/%d] path1 Loss: %.2f A_1: %.2f B_2: %.2f re: %.2f" %\
                (i+1, epoches, r_loss_1_path1+r_loss_2_path1,r_loss_1_path1,r_loss_2_path1,train_re_loss))
            print("[%d/%d] path2 Loss: %.2f A_2: %.2f B_1: %.2f re: %.2f" %\
                (i+1, epoches, r_loss_1_path2+r_loss_2_path2,r_loss_2_path2,r_loss_1_path2,train_re_loss))
            
