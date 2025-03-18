import os
import numpy as np
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from test3_2classify import read_data_physionet_4_with_val
from net1d import Net1D, MyDataset
from sklearn.model_selection import StratifiedKFold
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
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score




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

def run_exp(base_filters, filter_list, m_blocks_list, n_splits=10):

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    # 设置日志文件路径
    log_file = "training_log.txt"
    logger = setup_logger(log_file)

    # 获取数据
    X_train_val, X_test, Y_train_val, Y_test, pid_train_val, pid_test = read_data_physionet_4_with_val()
    
    # 初始化TensorBoard记录器
    writer = SummaryWriter(os.path.join(os.getcwd(), 'cross_validation-1'))

        # 初始化整体混淆矩阵
    n_classes = len(np.unique(Y_train_val))  # 假设 Y_train_val 包含所有类别
    overall_conf_matrix = np.zeros((n_classes, n_classes), dtype=int)

    # 10折交叉验证
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold = 0

    for train_idx, val_idx in kf.split(X_train_val ,Y_train_val):
        fold += 1
        print(f"Processing fold {fold}/{n_splits}")

        # 分割训练集和验证集
        X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
        Y_train, Y_val = Y_train_val[train_idx], Y_train_val[val_idx]
        pid_train, pid_val = pid_train_val[train_idx], pid_train_val[val_idx]


        # 计算并打印每个折的类别比例
        train_class_counts = Counter(Y_train)
        val_class_counts = Counter(Y_val)
        test_class_counts = Counter(Y_test)

        # 格式化类别比例，确保按类别编号（0~3）排序
        train_class_distribution = {i: train_class_counts[i] for i in range(4)}
        val_class_distribution = {i: val_class_counts[i] for i in range(4)}
        test_class_distribution = {i: test_class_counts[i] for i in range(4)}

        # 打印类别比例
        print(f"Fold {fold} - Train Class Distribution: {train_class_distribution}")
        print(f"Fold {fold} - Val Class Distribution: {val_class_distribution}")
        print(f"Fold {fold} - Test Class Distribution: {test_class_distribution}")

        # 创建数据集和数据加载器
        dataset = MyDataset(X_train, Y_train)
        dataset_val = MyDataset(X_val, Y_val)
        dataset_test = MyDataset(X_test, Y_test)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, drop_last=False)
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size, drop_last=False)

        # 创建模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            n_classes=2)
        model.to(device)

        # 定义优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        loss_func = torch.nn.CrossEntropyLoss()

        # 训练和验证
        n_epoch = 50
        for epoch in tqdm(range(n_epoch), desc=f"Epoch", leave=False):
            model.train()
            train_loss = 0.0
            train_pred, train_gt = [], []
            for batch_idx, batch in enumerate(dataloader):
                input_x, input_y = tuple(t.to(device) for t in batch)
                pred = model(input_x)
                loss = loss_func(pred, input_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_pred.append(pred.argmax(dim=1).cpu().numpy())
                train_gt.append(input_y.cpu().numpy())

            train_loss /= len(dataloader)
            train_pred = np.concatenate(train_pred)
            train_gt = np.concatenate(train_gt)
            train_accuracy, train_precision, train_recall = compute_metrics(train_gt, train_pred, average='macro')


            # 验证
            model.eval()
            val_loss = 0.0
            val_pred, val_gt = [], []
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader_val):
                    input_x, input_y = tuple(t.to(device) for t in batch)
                    pred = model(input_x)
                    loss = loss_func(pred, input_y)
                    val_loss += loss.item()
                    val_pred.append(pred.argmax(dim=1).cpu().numpy())
                    val_gt.append(input_y.cpu().numpy())

            val_loss /= len(dataloader_val)
            val_pred = np.concatenate(val_pred)
            val_gt = np.concatenate(val_gt)
            val_accuracy, val_precision, val_recall = compute_metrics(val_gt, val_pred, average='macro')

            # 计算当前折的混淆矩阵
            conf_matrix = confusion_matrix(val_gt, val_pred)
            # 累加到整体混淆矩阵
            overall_conf_matrix += conf_matrix

            scheduler.step(val_loss) 
            # 测试
            test_loss = 0.0
            test_pred, test_gt = [], []
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader_test):
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

            # 记录到TensorBoard
            writer.add_scalar(f'Fold_{fold}/Train_Loss', train_loss, epoch)
            writer.add_scalar(f'Fold_{fold}/Train_Accuracy', train_accuracy, epoch)
            writer.add_scalar(f'Fold_{fold}/Train_Precision', train_precision, epoch)
            writer.add_scalar(f'Fold_{fold}/Train_Recall', train_recall, epoch)

            writer.add_scalar(f'Fold_{fold}/Val_Loss', val_loss, epoch)
            writer.add_scalar(f'Fold_{fold}/Val_Accuracy', val_accuracy, epoch)
            writer.add_scalar(f'Fold_{fold}/Val_Precision', val_precision, epoch)
            writer.add_scalar(f'Fold_{fold}/Val_Recall', val_recall, epoch)

            writer.add_scalar(f'Fold_{fold}/Test_Loss', test_loss, epoch)
            writer.add_scalar(f'Fold_{fold}/Test_Accuracy', test_accuracy, epoch)
            writer.add_scalar(f'Fold_{fold}/Test_Precision', test_precision, epoch)
            writer.add_scalar(f'Fold_{fold}/Test_Recall', test_recall, epoch)

            # 记录混淆矩阵
            conf_matrix_val = confusion_matrix(val_gt, val_pred)
            conf_matrix_test = confusion_matrix(test_gt, test_pred)
            plot_confusion_matrix(conf_matrix_val, classes=["Class 0", "Class 1"], epoch=epoch, writer=writer, stage=f"Fold_{fold}_Validation")
            plot_confusion_matrix(conf_matrix_test, classes=["Class 0", "Class 1"], epoch=epoch, writer=writer, stage=f"Fold_{fold}_Test")

            # 打印日志
            logger.info(f"Fold {fold}, Epoch {epoch + 1}/{n_epoch}")
            logger.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")
            logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}")

        # 记录每个折的指标
        accuracies.append(test_accuracy)
        precisions.append(test_precision)
        recalls.append(test_recall)
        f1_scores.append(test_recall)

    # 可视化整体混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(overall_conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=np.arange(n_classes), yticklabels=np.arange(n_classes))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Overall Confusion Matrix")
    plt.tight_layout()

    # 保存到 TensorBoard
    writer.add_figure("Overall/Confusion_Matrix", plt.gcf(), 0)
    plt.close()

    # 计算总体指标的平均值和标准差
    average_accuracy = np.mean(accuracies)
    average_precision = np.mean(precisions)
    average_recall = np.mean(recalls)
    average_f1 = np.mean(f1_scores)

    std_accuracy = np.std(accuracies)
    std_precision = np.std(precisions)
    std_recall = np.std(recalls)
    std_f1 = np.std(f1_scores)

    # 记录总体指标到日志
    logger.info(f"Overall Average Accuracy: {average_accuracy:.4f} (+/- {std_accuracy:.4f})")
    logger.info(f"Overall Average Precision: {average_precision:.4f} (+/- {std_precision:.4f})")
    logger.info(f"Overall Average Recall: {average_recall:.4f} (+/- {std_recall:.4f})")
    logger.info(f"Overall Average F1 Score: {average_f1:.4f} (+/- {std_f1:.4f})")


    writer.close()




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

    # if is_debug:
    #     writer = SummaryWriter(os.path.join(os.getcwd(), 'debug--CPET-7'))
    # else:
    #     writer = SummaryWriter(os.path.join(os.getcwd(), 'first--CPET-7'))


   # 这个变量设置了基础过滤器的数量。在卷积神经网络中，过滤器（或称为卷积核）用于提取输入数据的特征。
    base_filters = 64
    # 这个列表定义了网络中每个卷积块或每个阶段之后输出的过滤器数量。
    # filter_list=[64,160,160,400,400,1024,1024]
    # # 这个列表定义了每个阶段中卷积块的数量。卷积块通常包含一个或多个卷积层，可能还包括激活函数、批量归一化层和池化层等。
    # m_blocks_list=[2,2,2,3,3,4,4]
    # filter_list=[64,160,400,1024]


    filter_list=[256,128,64,32]
    m_blocks_list=[2,2,2,2]



    run_exp(
        base_filters=base_filters,
        filter_list=filter_list,
        m_blocks_list=m_blocks_list)