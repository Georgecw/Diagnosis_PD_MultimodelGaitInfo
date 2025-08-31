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

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

class SingleHiddenLayerNN(nn.Module):
    """单隐藏层全连接神经网络"""
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.2):
        super(SingleHiddenLayerNN, self).__init__()
        
        # 输入层到隐藏层
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        # 隐藏层到输出层
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def load_data():
    """加载训练和测试数据"""
    print("正在加载数据...")
    
    # 加载训练数据
    X_train = pd.read_csv('neural network/X_train_reduced.csv')
    y_train = pd.read_csv('neural network/y_train.csv')
    
    # 加载测试数据
    X_test = pd.read_csv('neural network/X_test_reduced.csv')
    y_test = pd.read_csv('neural network/y_test.csv')
    
    # 移除空行
    X_train = X_train.dropna()
    y_train = y_train.dropna()
    X_test = X_test.dropna()
    y_test = y_test.dropna()
    
    print(f"训练数据形状: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"测试数据形状: X_test {X_test.shape}, y_test {y_test.shape}")
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

def train_model(model, train_loader, criterion, optimizer, num_epochs=200):
    """训练模型"""
    print(f"开始训练模型，共{num_epochs}个epoch...")
    
    train_losses = []
    train_accuracies = []
    
    model.train()
    for epoch in range(num_epochs):
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
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')
    
    return train_losses, train_accuracies

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

def plot_training_history(train_losses, train_accuracies):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    ax1.plot(train_losses)
    ax1.set_title('训练损失')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(train_accuracies)
    ax2.set_title('训练准确率')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('neural network/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.savefig('neural network/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 加载数据
    X_train, y_train, X_test, y_test = load_data()
    
    # 预处理数据
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, label_encoder = preprocess_data(
        X_train, y_train, X_test, y_test
    )
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # 模型参数
    input_size = X_train_tensor.shape[1]  # 29个特征
    hidden_size = 64  # 隐藏层神经元数量
    output_size = len(label_encoder.classes_)  # 4个类别
    
    print(f"模型架构: 输入层({input_size}) -> 隐藏层({hidden_size}) -> 输出层({output_size})")
    
    # 创建模型
    model = SingleHiddenLayerNN(input_size, hidden_size, output_size)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # 训练模型
    train_losses, train_accuracies = train_model(model, train_loader, criterion, optimizer, num_epochs=200)
    
    # 评估模型
    accuracy, f1, y_test_labels, predicted_labels = evaluate_model(
        model, X_test_tensor, y_test_tensor, label_encoder
    )
    
    # 可视化结果
    plot_training_history(train_losses, train_accuracies)
    plot_confusion_matrix(y_test_labels, predicted_labels, label_encoder.classes_)
    
    # 保存模型
    torch.save(model.state_dict(), 'neural network/model.pth')
    print("模型已保存为 'neural network/model.pth'")
    
    return model, accuracy, f1

if __name__ == "__main__":
    model, accuracy, f1 = main()
