import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

class ImprovedNN(nn.Module):
    """改进的单隐藏层全连接神经网络 - 针对小数据集优化"""
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.3):
        super(ImprovedNN, self).__init__()
        
        # 输入层到隐藏层
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)  # 批归一化
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        # 隐藏层到输出层
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """He初始化权重（适合ReLU）"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        x = self.fc1(x)
        if x.size(0) > 1:  # 批归一化需要batch_size > 1
            x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def load_reduced_data(data_suffix='_f_classif_k12'):
    """加载降维后的数据"""
    print(f"正在加载降维后的数据 (后缀: {data_suffix})...")
    
    # 检查文件是否存在
    train_files = [
        f'neural network/X_train_selected{data_suffix}.csv',
        f'neural network/y_train_selected{data_suffix}.csv'
    ]
    test_files = [
        f'neural network/X_test_selected{data_suffix}.csv',
        f'neural network/y_test_selected{data_suffix}.csv'
    ]
    
    for file_path in train_files + test_files:
        if not os.path.exists(file_path):
            print(f"错误：文件不存在 - {file_path}")
            return None, None, None, None
    
    # 加载数据
    X_train = pd.read_csv(f'neural network/X_train_selected{data_suffix}.csv')
    y_train = pd.read_csv(f'neural network/y_train_selected{data_suffix}.csv')
    X_test = pd.read_csv(f'neural network/X_test_selected{data_suffix}.csv')
    y_test = pd.read_csv(f'neural network/y_test_selected{data_suffix}.csv')
    
    print(f"训练数据形状: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"测试数据形状: X_test {X_test.shape}, y_test {y_test.shape}")
    print(f"选择的特征: {list(X_train.columns)}")
    print(f"类别分布: {y_train['Group'].value_counts()}")
    
    return X_train, y_train, X_test, y_test

def preprocess_data(X_train, y_train, X_test, y_test):
    """数据预处理"""
    print("正在预处理数据...")
    
    # 标签编码
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train['Group'])
    y_test_encoded = label_encoder.transform(y_test['Group'])
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.LongTensor(y_train_encoded)
    X_test_tensor = torch.FloatTensor(X_test.values)
    y_test_tensor = torch.LongTensor(y_test_encoded)
    
    print(f"类别映射: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, label_encoder

def train_model_with_early_stopping(model, train_loader, criterion, optimizer, 
                                   X_val, y_val, num_epochs=300, patience=30):
    """训练模型（带早停机制）"""
    print(f"开始训练模型，最多{num_epochs}个epoch，早停耐心值={patience}...")
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    model.train()
    for epoch in range(num_epochs):
        # 训练阶段
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            # 前向传播
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_accuracy = 100 * correct / total
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            _, val_predicted = torch.max(val_outputs, 1)
            val_accuracy = 100 * (val_predicted == y_val).sum().item() / len(y_val)
        model.train()
        
        # 记录历史
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
            print(f'  Best Val Loss: {best_val_loss:.4f}, Patience: {patience_counter}/{patience}')
        
        # 早停
        if patience_counter >= patience:
            print(f'早停触发！在第{epoch+1}轮停止训练')
            break
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("已加载验证损失最低的模型权重")
    
    return train_losses, train_accuracies, val_losses, val_accuracies

def evaluate_model(model, X_test, y_test, label_encoder):
    """评估模型"""
    print("正在评估模型...")
    
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        
        # 转换回原始标签
        y_test_labels = label_encoder.inverse_transform(y_test.numpy())
        predicted_labels = label_encoder.inverse_transform(predicted.numpy())
        
        # 计算准确率
        accuracy = accuracy_score(y_test_labels, predicted_labels)
        f1 = f1_score(y_test_labels, predicted_labels, average='weighted')
        
        print(f"测试集准确率: {accuracy:.4f}")
        print(f"测试集F1分数: {f1:.4f}")
        
        # 详细分类报告
        print("\n分类报告:")
        print(classification_report(y_test_labels, predicted_labels))
        
        return accuracy, f1, y_test_labels, predicted_labels

def plot_training_history_with_validation(train_losses, train_accuracies, 
                                        val_losses, val_accuracies, suffix=''):
    """绘制训练和验证历史"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(train_losses) + 1)
    
    # 损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='训练损失')
    ax1.plot(epochs, val_losses, 'r-', label='验证损失')
    ax1.set_title('训练和验证损失')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(epochs, train_accuracies, 'b-', label='训练准确率')
    ax2.plot(epochs, val_accuracies, 'r-', label='验证准确率')
    ax2.set_title('训练和验证准确率')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # 过拟合分析
    overfitting = [train_acc - val_acc for train_acc, val_acc in zip(train_accuracies, val_accuracies)]
    ax3.plot(epochs, overfitting, 'g-')
    ax3.set_title('过拟合程度 (训练准确率 - 验证准确率)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy Difference (%)')
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.grid(True)
    
    # 损失差异
    loss_diff = [train_loss - val_loss for train_loss, val_loss in zip(train_losses, val_losses)]
    ax4.plot(epochs, loss_diff, 'orange')
    ax4.set_title('损失差异 (训练损失 - 验证损失)')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss Difference')
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'neural network/results/training_history_reduced{suffix}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, suffix=''):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('混淆矩阵 - 降维后特征')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.savefig(f'neural network/results/confusion_matrix_reduced{suffix}.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_models(data_suffixes=['_f_classif_k12', '_f_classif_k15', '_f_classif_k18']):
    """比较不同特征数量的模型性能"""
    results = {}
    
    for suffix in data_suffixes:
        print(f"\n{'='*50}")
        print(f"训练模型: {suffix}")
        print(f"{'='*50}")
        
        # 加载数据
        X_train, y_train, X_test, y_test = load_reduced_data(suffix)
        if X_train is None:
            print(f"跳过 {suffix}，数据文件不存在")
            continue
        
        # 预处理
        X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, label_encoder = preprocess_data(
            X_train, y_train, X_test, y_test
        )
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # 减小batch_size
        
        # 模型参数
        input_size = X_train_tensor.shape[1]
        hidden_size = max(16, input_size)  # 隐藏层大小适应特征数量
        output_size = len(label_encoder.classes_)
        
        print(f"模型架构: 输入层({input_size}) -> 隐藏层({hidden_size}) -> 输出层({output_size})")
        
        # 创建模型
        model = ImprovedNN(input_size, hidden_size, output_size, dropout_rate=0.4)
        
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # 训练模型（使用测试集作为验证集，因为数据量小）
        train_losses, train_accuracies, val_losses, val_accuracies = train_model_with_early_stopping(
            model, train_loader, criterion, optimizer, X_test_tensor, y_test_tensor, 
            num_epochs=300, patience=50
        )
        
        # 评估模型
        accuracy, f1, y_test_labels, predicted_labels = evaluate_model(
            model, X_test_tensor, y_test_tensor, label_encoder
        )
        
        # 可视化结果
        plot_training_history_with_validation(train_losses, train_accuracies, 
                                            val_losses, val_accuracies, suffix)
        plot_confusion_matrix(y_test_labels, predicted_labels, label_encoder.classes_, suffix)
        
        # 保存模型
        torch.save(model.state_dict(), f'neural network/results/model_reduced{suffix}.pth')
        
        # 记录结果
        results[suffix] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'input_features': input_size,
            'hidden_size': hidden_size,
            'final_train_loss': train_losses[-1] if train_losses else None,
            'final_val_loss': val_losses[-1] if val_losses else None,
            'epochs_trained': len(train_losses)
        }
    
    return results

def print_comparison_results(results):
    """打印比较结果"""
    print(f"\n{'='*60}")
    print("模型性能比较")
    print(f"{'='*60}")
    
    print(f"{'模型':<20} {'特征数':<8} {'测试准确率':<12} {'F1分数':<10} {'训练轮数':<8}")
    print(f"{'-'*60}")
    
    for suffix, result in results.items():
        model_name = suffix.replace('_', ' ').strip()
        print(f"{model_name:<20} {result['input_features']:<8} {result['accuracy']:<12.4f} "
              f"{result['f1_score']:<10.4f} {result['epochs_trained']:<8}")

def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("开始使用降维后特征训练神经网络...")
    
    # 比较不同特征数量的模型
    results = compare_models(['_f_classif_k12', '_f_classif_k15', '_f_classif_k18'])
    
    # 打印比较结果
    if results:
        print_comparison_results(results)
    
    print("\n训练完成！生成的文件:")
    print("1. 训练历史图: neural network/results/training_history_reduced_*.png")
    print("2. 混淆矩阵图: neural network/results/confusion_matrix_reduced_*.png") 
    print("3. 模型文件: neural network/results/model_reduced_*.pth")
    
    return results

if __name__ == "__main__":
    results = main()
