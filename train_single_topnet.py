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

train_noisy_data = False # train noisy data
save_model_base_dir = './saved_models'
# save_model_dir = 'Nov18_single_topnet_models'
# save_model_dir = 'Nov26_single_topnet_vanilla_models'
save_model_dir = 'May12_single_topnet_models_noise_10_20_30_40'
learning_rate = 0.001
bs = 128
epoches = 100
partial_point_cloud_number = 512
coarse_num = 2048


# 1,3,5,6,8,9
for t in [1,3,5,6,8,9]:
    for split_num in range(5):
        for path in [1,2]:
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
            print("path%d"%path)
            print("fold%d"%valid_split[0])


            if path==1:
                single_topnet = topnet.TopNet_path1(partial_point_cloud_number, coarse_num)
            else:
                single_topnet = topnet.TopNet_path2(partial_point_cloud_number, coarse_num)
            single_topnet.cuda()

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
            optimizer = torch.optim.Adam(single_topnet.parameters(), lr=learning_rate)

            min_loss = 1e7
            best_epoch = -1
            for i in range(epoches):
                r_loss_1 = []
                r_loss_2 = []
                for AB_partial, gt_coarse in dataloader_train:
                    AB_partial = Variable(AB_partial).permute(0,2,1)
                    gt_coarse = Variable(gt_coarse).permute(0,2,1)
                    AB_partial, gt_coarse = rotate_2pc(AB_partial, gt_coarse)
                    AB_partial, gt_coarse = jitter_2pc(AB_partial, gt_coarse)
                    AB_partial, gt_coarse = scale_2pc(AB_partial, gt_coarse)
                    AB_partial = Variable(AB_partial).cuda().permute(0,2,1)
                    gt_coarse = Variable(gt_coarse).cuda()
                    optimizer.zero_grad()

                    if path==1:
                        A_1, B_2 = single_topnet(AB_partial)
                        A_1 = A_1.permute(0,2,1)
                        B_2 = B_2.permute(0,2,1)
                        loss_A_1 = torch.mean(EMD(A_1, gt_coarse[:,0:int(coarse_num/2),:], 0.005, 50)[0])
                        loss_B_2 = torch.mean(EMD(B_2, gt_coarse[:,int(coarse_num/2):,:], 0.005, 50)[0])
                        loss = loss_A_1 + loss_B_2
                        loss.backward()
                        optimizer.step()

                        r_loss_1.append(np.sqrt(loss_A_1.item())*100)
                        r_loss_2.append(np.sqrt(loss_B_2.item())*100)
                    else:
                        B_1, A_2 = single_topnet(AB_partial)
                        B_1 = B_1.permute(0,2,1)
                        A_2 = A_2.permute(0,2,1)

                        loss_B_1 = torch.mean(EMD(B_1, gt_coarse[:,int(coarse_num/2):,:], 0.005, 50)[0])
                        loss_A_2 = torch.mean(EMD(A_2, gt_coarse[:,0:int(coarse_num/2),:], 0.005, 50)[0])
                        loss = loss_A_2 + loss_B_1
                        loss.backward()
                        optimizer.step()

                        r_loss_1.append(np.sqrt(loss_B_1.item())*100)
                        r_loss_2.append(np.sqrt(loss_A_2.item())*100)

                r_loss_1 = sum(r_loss_1)/len(r_loss_1)
                r_loss_2 = sum(r_loss_2)/len(r_loss_2)

                if path==1:
                    print("[%d/%d] Loss: %.2f A_1: %.2f B_2: %.2f" %\
                        (i+1, epoches, r_loss_1+r_loss_2,r_loss_1,r_loss_2))
                else:
                    print("[%d/%d] Loss: %.2f B_1: %.2f A_2: %.2f" %\
                        (i+1, epoches, r_loss_1+r_loss_2,r_loss_1,r_loss_2))
                if (i+1)%5==0:
                    dir_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(dir_name)
                    
                    v_loss_1 = []
                    v_loss_2 = []
                    for AB_partial, gt_coarse in dataloader_valid:
                        AB_partial = Variable(AB_partial).permute(0,2,1)
                        gt_coarse = Variable(gt_coarse).permute(0,2,1)
                        AB_partial, gt_coarse = rotate_2pc(AB_partial, gt_coarse)
                        AB_partial = Variable(AB_partial).cuda().permute(0,2,1)
                        gt_coarse = Variable(gt_coarse).cuda()

                        if path==1:
                            A_1, B_2 = single_topnet(AB_partial)
                            A_1 = A_1.permute(0,2,1)
                            B_2 = B_2.permute(0,2,1)

                            loss_A_1 = torch.mean(EMD(A_1, gt_coarse[:,0:int(coarse_num/2),:], 0.005, 50)[0])
                            loss_B_2 = torch.mean(EMD(B_2, gt_coarse[:,int(coarse_num/2):,:], 0.005, 50)[0])

                            v_loss_1.append(np.sqrt(loss_A_1.item())*100)
                            v_loss_2.append(np.sqrt(loss_B_2.item())*100)
                        elif path==2:
                            B_1, A_2 = single_topnet(AB_partial)
                            B_1 = B_1.permute(0,2,1)
                            A_2 = A_2.permute(0,2,1)

                            loss_B_1 = torch.mean(EMD(B_1, gt_coarse[:,int(coarse_num/2):,:], 0.005, 50)[0])
                            loss_A_2 = torch.mean(EMD(A_2, gt_coarse[:,0:int(coarse_num/2),:], 0.005, 50)[0])
                            
                            v_loss_1.append(np.sqrt(loss_B_1.item())*100)
                            v_loss_2.append(np.sqrt(loss_A_2.item())*100)

                    v_loss_1 = sum(v_loss_1)/len(v_loss_1)
                    v_loss_2 = sum(v_loss_2)/len(v_loss_2)
                    if path==1:
                        print("------------------------------------------------------------------------")
                        print("valid results:")
                        print("[%d/%d] Loss: %.2f A_1: %.2f B_2: %.2f" %\
                            (i+1, epoches, v_loss_1+v_loss_2,v_loss_1,v_loss_2))
                        print("------------------------------------------------------------------------")
                    else:
                        print("------------------------------------------------------------------------")
                        print("valid results:")
                        print("[%d/%d] Loss: %.2f B_1: %.2f A_2: %.2f" %\
                            (i+1, epoches, v_loss_1+v_loss_2,v_loss_1,v_loss_2))
                        print("------------------------------------------------------------------------")
                    min_loss = min(v_loss_1+v_loss_2, min_loss)
                    if (v_loss_1+v_loss_2)==min_loss:
                        best_epoch = i+1
                        if os.path.exists(os.path.join('%s'%save_model_base_dir,'%s'%save_model_dir))==False:
                            os.mkdir(os.path.join('%s'%save_model_base_dir,'%s'%save_model_dir))
                        if os.path.exists(os.path.join('%s'%save_model_base_dir,'%s'%save_model_dir,'type%d'%t))==False:
                            os.mkdir(os.path.join('%s'%save_model_base_dir,'%s'%save_model_dir,'type%d'%t))
                        if os.path.exists(os.path.join('%s'%save_model_base_dir,'%s'%save_model_dir,'type%d'%t,'fold%d'%valid_split[0]))==False:
                            os.mkdir(os.path.join('%s'%save_model_base_dir,'%s'%save_model_dir,'type%d'%t,'fold%d'%valid_split[0]))
                        if path==1:
                            torch.save(single_topnet.state_dict(),'%s/%s/type%d/fold%d/best_single_topnet_path1_params.pth'%(save_model_base_dir, save_model_dir, t, valid_split[0]))
                        else:
                            torch.save(single_topnet.state_dict(),'%s/%s/type%d/fold%d/best_single_topnet_path2_params.pth'%(save_model_base_dir, save_model_dir, t, valid_split[0]))
                    print("best epoch: %d min loss: %.2f"%(best_epoch, min_loss))
