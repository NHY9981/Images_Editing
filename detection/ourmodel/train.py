from __future__ import print_function, division
import torch.optim as optim
import sys
import time
import numpy as np
import os
import torch
from torch.utils.data import WeightedRandomSampler
import torch.nn as nn
from torch.utils import data
from network.efficient_vit import EfficientViT   #替换成你自己的模型
from network.transform import train_transform, val_transform  # 替换成你自己的数据增强
from sklearn.metrics import accuracy_score
from random import shuffle
from network.utils import Dataset_Csv
import csv
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

# 在配置部分添加对抗训练参数
adversarial_config = {
    'enable_adv_train': True,      # 是否启用对抗训练
    'adv_train_ratio': 0.5,       # 对抗样本在训练中的比例 (0-1)
    'pgd_epsilon': 8/255,         # PGD攻击的扰动大小
    'pgd_alpha': 2/255,           # PGD单步扰动大小
    'pgd_iterations': 7,          # PGD迭代次数
    'adv_start_epoch': 3,         # 从第几个epoch开始对抗训练
    'adv_loss_weight': 0.5        # 对抗损失的权重
}


train_list = []
train_label = []
test_list = []
test_label = []
batch_size = 32 # 确保在这里或 __main__ 中定义
dataloaders = {}
log = None

current_training_epoch = 0 # 初始化

def get_current_epoch_for_dataset():
    global current_training_epoch
    return current_training_epoch

# PGD攻击类
class PGD_Attack:
    def __init__(self, model, epsilon=8/255, alpha=2/255, iters=7):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.iters = iters
    
    def attack(self, images, labels):
        ori_images = images.clone().detach()
        
        # 初始随机扰动
        images = images + torch.empty_like(images).uniform_(-self.epsilon, self.epsilon)
        images = torch.clamp(images, min=0, max=1).detach()
        
        for _ in range(self.iters):
            images.requires_grad = True
            outputs = self.model(images)
            
            # 计算损失
            loss = F.binary_cross_entropy_with_logits(outputs, labels)
            
            # 计算梯度
            grad = torch.autograd.grad(loss, images, 
                                     retain_graph=False, 
                                     create_graph=False)[0]
            
            # 更新对抗样本
            adv_images = images + self.alpha * grad.sign()
            eta = torch.clamp(adv_images - ori_images, 
                             min=-self.epsilon, 
                             max=self.epsilon)
            images = torch.clamp(ori_images + eta, min=0, max=1).detach()
            
        return images

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

def make_weights_for_balanced_classes(train_dataset_labels, stage='train'): # 修改：接收标签列表
    targets = torch.tensor(train_dataset_labels) # 直接使用传入的标签列表
    class_sample_count = torch.tensor(
        [(targets == t).sum() for t in torch.unique(targets, sorted=True)])
    # 防止除以零（如果某个类别样本数为0）
    class_sample_count[class_sample_count == 0] = 1e-6 # 用一个很小的值代替0
    weight = 1. / class_sample_count.float()
    samples_weight = torch.tensor([weight[t] for t in targets])
    return samples_weight


# 修改train_model函数以包含对抗训练
def train_model(model, model_dir, criterion, optimizer, scheduler, num_epochs=10, initial_epoch_val=0):
    global current_training_epoch
    
    # 初始化PGD攻击器
    pgd_attack = PGD_Attack(
        model,
        epsilon=adversarial_config['pgd_epsilon'],
        alpha=adversarial_config['pgd_alpha'],
        iters=adversarial_config['pgd_iterations']
    )
    
    best_logloss = 10.0
    best_epoch = 0
    
    for epoch in range(initial_epoch_val, num_epochs):
        current_training_epoch = epoch
        log.write(f"Epoch {epoch}/{num_epochs - 1} - Starting...\n")
        
        # 检查是否应该开始对抗训练
        use_adv_train = (adversarial_config['enable_adv_train'] and 
                        epoch >= adversarial_config['adv_start_epoch'])
        
        if use_adv_train:
            log.write(f"Adversarial Training ENABLED this epoch (Ratio: {adversarial_config['adv_train_ratio']})\n")
        
        best_test_auc = -10.0
        best_test_logloss = 10.0
        epoch_start = time.time()
        model_out_path = os.path.join(model_dir, str(epoch) + '_cross_eff_weight.ckpt')
        
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_loss_train = 0.0
            y_scores, y_trues = [], []
            
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs, labels = inputs.cuda(), labels.to(torch.float32).cuda()
                
                if phase == 'train':
                    optimizer.zero_grad()
                    
                    # 标准训练
                    outputs = model(inputs)
                    if outputs.ndim == 1 or outputs.shape[1] == 1:
                        labels = labels.view_as(outputs)
                    loss = criterion(outputs, labels)
                    
                    # 对抗训练
                    adv_loss = 0
                    if use_adv_train and torch.rand(1) < adversarial_config['adv_train_ratio']:
                        # 生成对抗样本
                        adv_inputs = pgd_attack.attack(inputs, labels)
                        
                        # 计算对抗损失
                        adv_outputs = model(adv_inputs)
                        adv_loss = criterion(adv_outputs, labels)
                        
                        # 组合损失
                        total_loss = (1 - adversarial_config['adv_loss_weight']) * loss + \
                                    adversarial_config['adv_loss_weight'] * adv_loss
                    else:
                        total_loss = loss
                    
                    total_loss.backward()
                    optimizer.step()
                    
                    # 记录损失
                    batch_loss = total_loss.data.item()
                    preds = torch.sigmoid(outputs)
                else:
                    with torch.no_grad():
                        outputs = model(inputs)
                        if outputs.ndim == 1 or outputs.shape[1] == 1:
                            labels = labels.view_as(outputs)
                        loss = criterion(outputs, labels)
                        batch_loss = loss.data.item()
                        preds = torch.sigmoid(outputs)
                
                running_loss += batch_loss
                running_loss_train += batch_loss
                
                y_true = labels.data.cpu().numpy()
                y_score = preds.data.cpu().numpy()

                if i % 100 == 0:
                    # 处理 y_true 和 y_score 的形状，确保它们是1D的
                    y_true_flat = y_true.ravel()
                    y_score_flat = y_score.ravel()
                    batch_acc = accuracy_score(y_true_flat, np.where(y_score_flat > 0.5, 1, 0))
                    log.write(
                        'Epoch {}/{} Batch {}/{} Stage: {} Logloss: {:.4f} Accuracy: {:.4f}\n'.format(epoch,
                                                                                                      num_epochs - 1,
                                                                                                      i, len(
                                dataloaders[phase]), phase, batch_loss, batch_acc))
                if (i + 1) % 500 == 0 and phase == 'train': # 只在训练阶段的特定间隔进行验证和保存
                    inter_loss = running_loss_train / 500.0
                    log.write('last phase train loss is {}\n'.format(inter_loss))
                    running_loss_train = 0.0
                    test_loss = val_models(model, criterion, num_epochs, test_list, epoch) # test_list 需要在 main 中定义
                    if test_loss < best_test_logloss:
                        best_test_logloss = test_loss
                        log.write('save current model {}, Now time is {}, best logloss is {}\n'.format(i,time.asctime( time.localtime(time.time()) ),best_test_logloss))
                        model_out_paths = os.path.join(model_dir, str(epoch) + "_" + str(i) + '_test_best_weight.ckpt') # 文件名修改
                        torch.save(model.module.state_dict(), model_out_paths)
                    model.train() # 确保模型返回训练模式
                    if scheduler is not None:
                         log.write('now lr is : {}\n'.format(scheduler.get_last_lr())) # 使用 get_last_lr()

                if phase == 'test': # 在测试阶段累积所有预测
                    y_scores.extend(y_score.ravel()) # 确保是1D
                    y_trues.extend(y_true.ravel())  # 确保是1D

            if phase == 'train':
                if scheduler is not None: # 添加None检查
                    scheduler.step()
            
            if phase == 'test': # 在测试阶段结束后计算总的 loss 和 accuracy
                epoch_loss = running_loss / (len(test_list) / batch_size) if len(test_list) > 0 else 0 # 避免除以零
                y_trues_arr, y_scores_arr = np.array(y_trues), np.array(y_scores)
                if len(y_trues_arr) > 0: # 确保有数据点
                    accuracy = accuracy_score(y_trues_arr, np.where(y_scores_arr > 0.5, 1, 0))
                    log.write(
                        '**Epoch {}/{} Stage: {} Logloss: {:.4f} Accuracy: {:.4f}\n'.format(epoch, num_epochs - 1, phase,
                                                                                            epoch_loss,
                                                                                            accuracy))
                else:
                    log.write(f'**Epoch {epoch}/{num_epochs-1} Stage: {phase} No test data processed.\n')


            if phase == 'test' and epoch_loss < best_logloss and len(y_trues_arr)>0 : # 添加检查
                best_logloss = epoch_loss
                best_epoch = epoch
                torch.save(model.module.state_dict(), model_out_path) # 保存每个 epoch 结束时最好的模型

            
                
        log.write('Epoch {}/{} Time {}s\n'.format(epoch, num_epochs - 1, time.time() - epoch_start))
    log.write('***************************************************\n') # 修正日志格式
    log.write('Best logloss {:.4f} and Best Epoch is {}\n'.format(best_logloss, best_epoch))


def val_models(model, criterion, num_epochs, test_list_paths, current_epoch=0 ,phase='test'): # 参数名修改
    log.write('------------------------------------------------------------------------\n')
    model.eval()
    running_loss_val = 0.0
    y_scores, y_trues = [], []
    for k, (inputs_val, labels_val) in enumerate(dataloaders[phase]): # 假设 dataloaders['test'] 已正确设置
        inputs_val, labels_val = inputs_val.cuda(), labels_val.to(torch.float32).cuda()
        with torch.no_grad():
            outputs_val = model(inputs_val)
            if outputs_val.ndim == 1 or outputs_val.shape[1] == 1:
                 labels_val = labels_val.view_as(outputs_val)
            loss = criterion(outputs_val, labels_val)
            preds = torch.sigmoid(outputs_val)
        batch_loss = loss.data.item()
        running_loss_val += batch_loss

        y_true = labels_val.data.cpu().numpy().ravel() #确保1D
        y_score = preds.data.cpu().numpy().ravel() #确保1D

        if k % 100 == 0:
            batch_acc = accuracy_score(y_true, np.where(y_score > 0.5, 1, 0))
            log.write(
                'Validation Epoch {}/{} Batch {}/{} Stage: {} Logloss: {:.4f} Accuracy: {:.4f}\n'.format(current_epoch, # 日志信息调整
                                                                                              num_epochs - 1,
                                                                                              k, len(dataloaders[phase]),
                                                                                              phase, batch_loss, batch_acc))
        y_scores.extend(y_score)
        y_trues.extend(y_true)

    epoch_loss = running_loss_val / (len(test_list_paths) / batch_size) if len(test_list_paths) > 0 else 0 # 参数名修改 & 避免除以零
    y_trues_arr, y_scores_arr = np.array(y_trues), np.array(y_scores)

    if len(y_trues_arr) > 0:
        accuracy = accuracy_score(y_trues_arr, np.where(y_scores_arr > 0.5, 1, 0))
        
        # ============= AUC 计算 =============
        try:
            auc = roc_auc_score(y_trues_arr, y_scores_arr)
        except ValueError as e:
            auc = -1  # 如果计算失败（如只有单一类别）
            log.write(f"AUC calculation failed: {str(e)}\n")
        # ========================================
        
        log.write(
            '**Validation Epoch {}/{} Stage: {} Logloss: {:.4f} Accuracy: {:.4f} AUC: {:.4f}\n'.format(
                current_epoch, num_epochs - 1, phase, 
                epoch_loss, accuracy, auc
            ))
        
        try: # 添加 try-except 处理 confusion_matrix 可能的错误
            tn, fp, fn, tp = confusion_matrix(y_trues_arr, np.where(y_scores_arr > 0.5, 1, 0)).ravel()
            # 避免除以零
            tnr = tn/(fp + tn) if (fp + tn) > 0 else 0
            fpr = fp/(fp + tn) if (fp + tn) > 0 else 0
            fnr = fn/(tp + fn) if (tp + fn) > 0 else 0
            tpr = tp/(tp + fn) if (tp + fn) > 0 else 0
            log.write(
                '**Validation Epoch {}/{} Stage: {} TNR: {:.2f} FPR: {:.2f} FNR: {:.2f} TPR: {:.2f} \n'.format(current_epoch, num_epochs - 1, phase,
                                                                                    tnr, fpr, fnr, tpr))
        except ValueError as e:
            log.write(f"Could not compute confusion matrix for validation epoch {current_epoch}: {e}\n")
    else:
        log.write(f'**Validation Epoch {current_epoch}/{num_epochs-1} Stage: {phase} No validation data processed.\n')

    log.write('***************************************************\n')
    return epoch_loss

def base_data(csv_file, path_list_ref, label_list_ref): # 修改：通过引用传递列表
    frame_reader = open(csv_file, 'r')
    csv_reader = csv.reader(frame_reader)
    for f in csv_reader:
        path = f[0]
        label = int(f[1])
        label_list_ref.append(label)
        path_list_ref.append(path)
    frame_reader.close() # 关闭文件
    print(f'Loaded {len(label_list_ref)} training samples from {csv_file}') # 更详细的日志
    log.write(f'Loaded {len(path_list_ref)} training samples from {csv_file}\n') # 更详细的日志

def validation_data(csv_file, path_list_ref, label_list_ref): # 修改：通过引用传递列表
    frame_reader = open(csv_file, 'r')
    fnames = csv.reader(frame_reader)
    for f in fnames:
        path = f[0]
        label = int(f[1])
        label_list_ref.append(label)
        path_list_ref.append(path)
    frame_reader.close()
    log.write(f'Loaded {len(label_list_ref)} validation samples from {csv_file}\n') # 更详细的日志


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    start = time.time()
    current_epoch = 0
    train_csv = "/root/autodl-tmp/csv/dfgc_train_balanced.csv"
    val_csv = "/root/autodl-tmp/csv/dfgc_val.csv"

    model_dir = '/root/autodl-tmp/cross_eff'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    log_name = model_dir.split('/')[-1] + '.log' 
    log_dir = os.path.join(model_dir, log_name)
    if os.path.exists(log_dir) and current_epoch == 0: # 只在从头开始训练时删除
        os.remove(log_dir)
        print('The log file is exit!') # 修正: "The log file existed and was removed!"

    log = Logger(log_dir, sys.stdout)
    model_name = model_dir.split('/')[-1]
    log.write(f'model :  {model_name}  batch_size : {batch_size} frames : 10 \n') # 使用 f-string
    
    # 对抗训练配置 (用户可以修改这些参数)
    adversarial_config = {
        'enable_adv_train': True,      # 是否启用对抗训练
        'adv_train_ratio': 0.5,        # 对抗样本在训练中的比例 (0-1)
        'pgd_epsilon': 8/255,          # PGD攻击的扰动大小
        'pgd_alpha': 2/255,            # PGD单步扰动大小
        'pgd_iterations': 7,           # PGD迭代次数
        'adv_start_epoch': 3,          # 从第几个epoch开始对抗训练
        'adv_loss_weight': 0.5         # 对抗损失的权重
    }
    
    log.write("\nAdversarial Training Configuration:\n")
    for k, v in adversarial_config.items():
        log.write(f"{k}: {v}\n")
    
    # MODIFICATION: 记录是否使用增强
    log.write('Data Augmentation for training: True\n')


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    log.write(f'Using device: {device}\n')

    params = {'shuffle': False, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}


    log.write('loading train data' + '\n')
    base_data(train_csv, train_list, train_label) # 假设 base_data 修改为接收列表引用

    if train_list: # 只有列表非空时 shuffle
        ziplist = list(zip(train_list, train_label))
        shuffle(ziplist)
        train_list[:], train_label[:] = zip(*ziplist)
    else:
        log.write("Warning: train_list is empty after loading.\n")


    log.write('loading val data' + '\n')
    validation_data(val_csv, test_list, test_label) 


    log.write('Creating Datasets with epoch-conditional augmentation...\n')
    augment_start_epoch = 5 # 指定从第几个 epoch 之后开始增强 

    if train_list:
        train_set = Dataset_Csv(
            folders=train_list,
            labels=train_label,
            transform=train_transform, # 传递整个字典
            should_augment=True,         # 仍然需要设置为 True 来启用增强逻辑
            augment_after_epoch=augment_start_epoch,
            get_current_epoch_fn=get_current_epoch_for_dataset # 传递获取 epoch 的函数
        )
    else:
        train_set = None
        log.write("Warning: train_list is empty. Cannot create training dataset.\n")

    if test_list:
        valid_set = Dataset_Csv(
            folders=test_list,
            labels=test_label,
            transform=val_transform, # 传递整个字典
            should_augment=False,        # 验证集通常不增强
        )
    else:
        valid_set = None
        log.write("Warning: test_list is empty. Cannot create validation dataset.\n")

    images_datasets_labels = {} # 重命名以更准确反映内容
    images_datasets_labels['train'] = train_label
    images_datasets_labels['test'] = test_label

    weights = {}
    data_sampler = {}

    if train_label: # 确保 train_label 非空
        weights['train'] = make_weights_for_balanced_classes(images_datasets_labels['train'], stage='train')
        data_sampler['train'] = WeightedRandomSampler(weights['train'], len(images_datasets_labels['train']), replacement=True)
        # MODIFICATION: 将 dataloaders 赋值给全局变量（如果 train_model 依赖它）
        dataloaders['train'] = data.DataLoader(train_set, sampler=data_sampler['train'], batch_size=batch_size, **params)
    else:
        log.write("Warning: train_label is empty. Cannot create weighted sampler. Trying standard DataLoader if train_set exists.\n")
        if train_set and len(train_set) > 0:
             dataloaders['train'] = data.DataLoader(train_set, batch_size=batch_size, **params)
        else:
             dataloaders['train'] = None
             log.write("Error: train_set is also empty or not created. Exiting.\n")
             # sys.exit(1) # 如果没有训练数据，可能需要退出


    if test_label: # 确保 test_label 非空
        # 验证集通常不需要加权采样，除非类别极度不平衡
        dataloaders['test'] = data.DataLoader(valid_set, batch_size=batch_size, **params) # 验证集不 shuffle
    else:
        log.write("Warning: test_label is empty. Test DataLoader will be None.\n")
        dataloaders['test'] = None


    # datasets_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']} # image_datasets 未完全定义
    datasets_sizes = {x: len(dataloaders[x]) if dataloaders[x] is not None else 0 for x in ['train', 'test']}
    log.write(f'Dataset sizes (number of batches): {datasets_sizes}\n')

    if dataloaders['train'] is None:
        log.write("FATAL: Training dataloader is None. Cannot proceed with training.\n")
        sys.exit(1)
 
    model = EfficientViT()

    # Define parameter groups for differential learning rates
    param_groups = [
        {
            'params': model.cnn_backbone.parameters(),
            'lr': 1e-5  # For the pretrained CNN backbone
        },
        {
            # Parameters for the projection, CLS token, positional embeddings, and Transformer encoder
            'params': list(model.linear_proj.parameters()) + \
                    [model.cls_token] + \
                    [model.pos_embed] + \
                    list(model.transformer_encoder.parameters()),
            'lr': 5e-6  # For the "ViT" like components
        },
        {
            # Parameters for the final normalization layer and the classification head
            'params': list(model.norm.parameters()) + \
                    list(model.head.parameters()),
            'lr': 1e-4  # For the classification head and its preceding norm
        }
    ]

    # Create the AdamW optimizer
    optimizer_ft = optim.AdamW(param_groups, weight_decay=1e-4)
    model = nn.DataParallel(model.to(device)) # 将模型移到 device
    criterion = nn.BCEWithLogitsLoss()
    num_epochs = 10
    from torch.optim.lr_scheduler import CosineAnnealingLR
    # 余弦退火调度
    exp_lr_scheduler = CosineAnnealingLR(optimizer_ft, T_max=20, eta_min=1e-7)


    train_model(model=model, model_dir=model_dir, criterion=criterion, optimizer=optimizer_ft,
                scheduler=exp_lr_scheduler,
                num_epochs=10, # 总共训练的 epochs
                initial_epoch_val=current_epoch) # 从这个 epoch 值开始循环

    elapsed = (time.time() - start)
    log.write('Total time is {}.\n'.format(elapsed))