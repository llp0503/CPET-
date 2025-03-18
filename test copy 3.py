import os
import numpy as np
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from test2 import read_data_physionet_4_with_val
from net1d import Net1D, MyDataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchsummary import summary
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
import logging


# 手动调整学习率的函数
def manual_lr_step(optimizer, epoch, start_epoch=20, end_epoch=30, initial_lr=1e-3, factor=0.1):
    """
    在指定的 epoch 范围内手动调整学习率。
    :param optimizer: 优化器
    :param epoch: 当前 epoch
    :param start_epoch: 开始手动调整的 epoch
    :param end_epoch: 结束手动调整的 epoch
    :param initial_lr: 初始学习率
    :param factor: 学习率调整因子
    """
    if start_epoch <= epoch <= end_epoch:
        # 计算当前 epoch 的学习率
        new_lr = initial_lr * (factor ** ((epoch - start_epoch) // 5))  # 每隔 5 个 epoch 调整一次
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        print(f"Manually set learning rate to {new_lr} at epoch {epoch}")


# 添加早停策略类
class EarlyStopping:
    def __init__(self, patience=10, delta=0, save_path="best_model.pth"):
        """
        早停类，用于监控验证集性能并提前终止训练。
        :param patience: 验证集性能没有改善时的最大容忍轮数。
        :param delta: 验证集性能的最小改善阈值。
        :param save_path: 最佳模型权重的保存路径。
        """
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss, model):
        score = -val_loss  # 假设监控的是损失，损失越小越好

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0



    def save_checkpoint(self, model):
        """保存当前最佳模型权重"""
        torch.save(model.state_dict(), self.save_path)
        print(f"Model saved to {self.save_path}")

def get_current_lr(optimizer):
    """获取当前学习率"""
    return optimizer.param_groups[0]['lr']


def setup_logger(log_file):
    """
    设置日志记录器，将日志保存到指定文件。
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger



def compute_metrics(y_true, y_pred, average='macro'):
    """
    计算准确率、精确率和召回率。
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :param average: 计算方式（'macro', 'micro', 'weighted' 等）
    :return: 准确率、精确率、召回率
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    return accuracy, precision, recall


def plot_confusion_matrix(conf_matrix, classes, epoch, writer, stage="Validation"):
    """
    可视化混淆矩阵并保存到 TensorBoard。
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title(f"{stage} Confusion Matrix")
    # 保存到 TensorBoard
    writer.add_figure(f"{stage}/Confusion_Matrix", plt.gcf(), epoch)
    plt.close()

def run_exp(base_filters, filter_list, m_blocks_list):
    # 设置日志文件路径
    log_file = "training_log.txt"
    logger = setup_logger(log_file)

    X_train, X_val, X_test, Y_train, Y_val, Y_test, pid_train, pid_val, pid_test = read_data_physionet_4_with_val()
    dataset = MyDataset(X_train, Y_train)
    dataset_val = MyDataset(X_val, Y_val)
    dataset_test = MyDataset(X_test, Y_test)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, drop_last=False)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, drop_last=False)
    # drop_last=False 是 DataLoader 类中的一个参数。它控制在处理数据时是否丢弃最后一个批次（batch）中的不完整样本。
    # 如果你选择了 drop_last=False，即使最后一个批次只有少数几个样本，模型依然会处理这个批次。
    # make model

    device_str = "cuda"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")


# kernel_size=16：指定卷积核的大小为16。这意味着每个卷积操作将在输入数据上滑动一个大小为16的窗口。
    model = Net1D(
        in_channels=30, 
        base_filters=base_filters, 
        ratio=1.0, 
        filter_list=filter_list, 
        m_blocks_list=m_blocks_list, 
        kernel_size=16, 
        stride=2, 
        groups_width=16,
        verbose=False, 
        n_classes=4)
    model.to(device)


    print(model)
    summary(model, (X_train.shape[1], X_train.shape[2]), device=device_str)
    # summary 函数通常来自于 torchsummary 库，它用于显示模型的详细架构信息，包括每一层的输出形状、参数数量等。你在代码中调用 summary(model, (X_train.shape[1], X_train.shape[2]), device=device_str)，意图是显示模型的摘要。
    
    
    # train and test
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    # 初始化了一个 torch.optim.lr_scheduler.ReduceLROnPlateau 调度器，其模式设置为 'min'。这意味着调度器会根据传入的标量值 越小越好 来调整学习率。通常情况下，这个标量值是验证集的损失值。
    loss_func = torch.nn.CrossEntropyLoss()
    # 早停策略
    # early_stopping = EarlyStopping(patience=10, delta=0.001, save_path="best_model.pth")
    n_epoch = 50
    step = 0
    # step 变量常用于作为计数器。例如，在训练循环中，它可能用来跟踪当前的训练步骤。
    for _ in tqdm(range(n_epoch), desc="epoch", leave=False):


        # manual_lr_step(optimizer, _, start_epoch=20, end_epoch=30, initial_lr=1e-4, factor=0.1)
        current_lr = get_current_lr(optimizer)
        writer.add_scalar('Learning_Rate', current_lr, _)
        # train
        model.train()
        train_loss = 0.0
        train_pred = []
        train_gt = []
        prog_iter = tqdm(dataloader, desc="Training", leave=False)
        # batch_idx 是当前批次的索引。
        # batch 是当前批次的数据，通常是一个元组，包含输入数据和标签。
        for batch_idx, batch in enumerate(prog_iter):

            input_x, input_y = tuple(t.to(device) for t in batch)
            pred = model(input_x)
            loss = loss_func(pred, input_y)
            optimizer.zero_grad()
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_pred.append(pred.argmax(dim=1).cpu().numpy())
            train_gt.append(input_y.cpu().numpy())
            writer.add_scalar('Loss/train', loss.item(), step)
            step += 1

            if is_debug:
                break

        train_loss /= len(dataloader)
        train_pred = np.concatenate(train_pred)
        train_gt = np.concatenate(train_gt)
        train_accuracy, train_precision, train_recall = compute_metrics(train_gt, train_pred, average='macro')
            

                    
        # val
        model.eval()
        val_loss = 0.0
        prog_iter_val = tqdm(dataloader_val, desc="Validation", leave=False)
        # 初始化两个列表，分别用于存储所有预测的概率和真实标签。
        val_pred = []
        val_gt = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(prog_iter_val):
                input_x, input_y = tuple(t.to(device) for t in batch)
                pred = model(input_x)
                val_pred.append(pred.argmax(dim=1).cpu().numpy())
                val_gt.append(input_y.cpu().numpy())
                loss = loss_func(pred, input_y)
                val_loss += loss.item()
        val_loss /= len(dataloader_val)
        scheduler.step(val_loss) 
        #         # 从第 30 轮开始，启用 ReduceLROnPlateau 调度器
        # if _ >= 20:
        #     scheduler.step(val_loss)   
        # all_pred_prob 和 all_gt：将列表中的预测概率和真实标签合并为 NumPy 数组。
        # all_pred：通过 np.argmax 获取预测的类别标签。
        # all_pred_prob = np.concatenate(all_pred_prob)
        val_loss /= len(dataloader_val)
        val_pred = np.concatenate(val_pred)
        val_gt = np.concatenate(val_gt)
        val_accuracy, val_precision, val_recall = compute_metrics(val_gt, val_pred, average='macro')


        # vote most common
        final_pred = []
        final_gt = []

        # Confusion Matrix
        conf_matrix_val = confusion_matrix(val_gt, val_pred)
        print(f"Epoch {_ + 1} Validation Confusion Matrix:\n", conf_matrix_val)
        plot_confusion_matrix(conf_matrix_val, classes=["Class 0", "Class 1", "Class 2", "Class 3"], epoch=_, writer=writer, stage="Validation")



        for i_pid in np.unique(pid_val):
            tmp_pred = val_pred[pid_val==i_pid]
            tmp_gt = Y_val[pid_val==i_pid]
            final_pred.append(Counter(tmp_pred).most_common(1)[0][0])
            final_gt.append(Counter(tmp_gt).most_common(1)[0][0])


        ## classification report
        tmp_report = classification_report(final_gt, final_pred, output_dict=True)
        print(confusion_matrix(final_gt, final_pred))
        f1_score = (tmp_report['0']['f1-score'] + tmp_report['1']['f1-score'] + tmp_report['2']['f1-score'] + tmp_report['3']['f1-score'])/4
        writer.add_scalar('F1/Validation/f1_score', f1_score, _)
        writer.add_scalar('F1/Validation/label_0', tmp_report['0']['f1-score'], _)
        writer.add_scalar('F1/Validation/label_1', tmp_report['1']['f1-score'], _)
        writer.add_scalar('F1/Validation/label_2', tmp_report['2']['f1-score'], _)
        writer.add_scalar('F1/Validation/label_3', tmp_report['3']['f1-score'], _)
                    
        # test
        model.eval()
        prog_iter_test = tqdm(dataloader_test, desc="Testing", leave=False)
        test_pred = []
        test_gt = []
        test_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(prog_iter_test):
                input_x, input_y = tuple(t.to(device) for t in batch)
                pred = model(input_x)
                loss = loss_func(pred, input_y)
                test_loss += loss.item()
                test_pred.append(pred.argmax(dim=1).cpu().numpy())
                test_gt.append(input_y.cpu().numpy())

        test_loss /= len(dataloader_test)
        test_pred = np.concatenate(test_pred)
        test_gt = np.concatenate(test_gt)
        test_accuracy, test_precision, test_recall = compute_metrics(test_gt, test_pred, average='macro')



        # # # # 早停检查
        # early_stopping(val_loss, model)
        # if early_stopping.early_stop:
        #     print("Early stopping triggered. Training stopped.")
        #     break


        # 记录日志
        logger.info(f"Epoch {_ + 1}/{n_epoch}")
        logger.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")
        logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}")


        # Confusion Matrix
        conf_matrix_test = confusion_matrix(test_gt, test_pred)
        print(f"Epoch {_+ 1} Test Confusion Matrix:\n", conf_matrix_test)
        plot_confusion_matrix(conf_matrix_test, classes=["Class 0", "Class 1", "Class 2", "Class 3"], epoch=_, writer=writer, stage="Test")
        
        
        ## vote most common
        final_pred = []
        final_gt = []
        for i_pid in np.unique(pid_test):
            tmp_pred = test_pred[pid_test==i_pid]
            tmp_gt = Y_test[pid_test==i_pid]
            final_pred.append(Counter(tmp_pred).most_common(1)[0][0])
            final_gt.append(Counter(tmp_gt).most_common(1)[0][0])

        ## classification report
        tmp_report = classification_report(final_gt, final_pred, output_dict=True)
        print(confusion_matrix(final_gt, final_pred))
        f1_score = (tmp_report['0']['f1-score'] + tmp_report['1']['f1-score'] + tmp_report['2']['f1-score'] + tmp_report['3']['f1-score'])/4
        writer.add_scalar('F1/Test/f1_score', f1_score, _)
        writer.add_scalar('F1/Test/label_0', tmp_report['0']['f1-score'], _)
        writer.add_scalar('F1/Test/label_1', tmp_report['1']['f1-score'], _)
        writer.add_scalar('F1/Test/label_2', tmp_report['2']['f1-score'], _)
        writer.add_scalar('F1/Test/label_3', tmp_report['3']['f1-score'], _)


    print(f"Epoch {_ + 1}/{n_epoch}")
    print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")




def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True)  # 有检查操作，看下文区别



if __name__ == "__main__":



 
    seed_torch()
    batch_size =32

    is_debug = False

    if is_debug:
        writer = SummaryWriter(os.path.join(os.getcwd(), 'debug--CPET-5'))
    else:
        writer = SummaryWriter(os.path.join(os.getcwd(), 'first--CPET-5'))


    # make data, (sample, channel, length)
    X_train, X_val, X_test, Y_train, Y_val, Y_test, pid_train, pid_val, pid_test = read_data_physionet_4_with_val()
    print(X_train.shape, Y_train.shape)

   # 这个变量设置了基础过滤器的数量。在卷积神经网络中，过滤器（或称为卷积核）用于提取输入数据的特征。
    base_filters = 64
    # 这个列表定义了网络中每个卷积块或每个阶段之后输出的过滤器数量。
    # filter_list=[64,160,160,400,400,1024,1024]
    # # 这个列表定义了每个阶段中卷积块的数量。卷积块通常包含一个或多个卷积层，可能还包括激活函数、批量归一化层和池化层等。
    # m_blocks_list=[2,2,2,3,3,4,4]
    # filter_list=[64,160,400,1024]


    filter_list=[256,128,64,32]

    # filter_list=[128,64,32,32]
    m_blocks_list=[2,2,2,2]



    run_exp(
        base_filters=base_filters,
        filter_list=filter_list,
        m_blocks_list=m_blocks_list)