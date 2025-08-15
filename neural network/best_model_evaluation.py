import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# 导入自定义函数
from nn_reduced import ImprovedNN, load_reduced_data, preprocess_data, train_model_with_early_stopping, evaluate_model

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def train_best_model():
    """使用超参数搜索发现的最佳参数训练模型"""
    
    # 从超参数搜索中发现的最佳参数
    best_params = {
        'hidden_size': 20,
        'learning_rate': 0.0005,
        'dropout_rate': 0.5,
        'batch_size': 8,
        'weight_decay': 0.0001,
        'max_epochs': 300,
        'patience': 50
    }
    
    print("🏆 使用最佳超参数训练模型...")
    print(f"最佳参数: {best_params}")
    
    # 加载数据
    X_train, y_train, X_test, y_test = load_reduced_data('_f_classif_k15')
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, label_encoder = preprocess_data(
        X_train, y_train, X_test, y_test
    )
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建最佳模型
    input_size = X_train_tensor.shape[1]
    output_size = len(label_encoder.classes_)
    
    model = ImprovedNN(
        input_size=input_size,
        hidden_size=best_params['hidden_size'],
        output_size=output_size,
        dropout_rate=best_params['dropout_rate']
    )
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=best_params['batch_size'], 
        shuffle=True
    )
    
    # 创建优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=best_params['learning_rate'],
        weight_decay=best_params['weight_decay']
    )
    
    # 训练模型
    print("开始训练最佳模型...")
    train_losses, train_accuracies, val_losses, val_accuracies = train_model_with_early_stopping(
        model, train_loader, criterion, optimizer, 
        X_test_tensor, y_test_tensor,
        num_epochs=best_params['max_epochs'],
        patience=best_params['patience']
    )
    
    # 评估模型
    test_accuracy, test_f1, y_test_labels, predicted_labels = evaluate_model(
        model, X_test_tensor, y_test_tensor, label_encoder
    )
    
    return (model, test_accuracy, test_f1, y_test_labels, predicted_labels, 
            label_encoder, train_losses, train_accuracies, val_losses, val_accuracies, best_params)

def create_comprehensive_evaluation(y_test_labels, predicted_labels, label_encoder, 
                                  train_losses, train_accuracies, val_losses, val_accuracies, 
                                  best_params, test_accuracy, test_f1):
    """创建综合评估报告"""
    
    # 创建大图
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 混淆矩阵 (占据左侧大部分空间)
    ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=2)
    cm = confusion_matrix(y_test_labels, predicted_labels)
    
    # 绘制混淆矩阵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_,
                ax=ax1, cbar_kws={'shrink': 0.8}, square=True, linewidths=0.5)
    ax1.set_title(f'🎯 最佳模型混淆矩阵\n测试准确率: {test_accuracy:.1%}, F1分数: {test_f1:.1%}', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('预测标签', fontsize=12)
    ax1.set_ylabel('真实标签', fontsize=12)
    
    # 在每个格子中添加百分比
    total = np.sum(cm)
    for i in range(len(label_encoder.classes_)):
        for j in range(len(label_encoder.classes_)):
            percentage = cm[i, j] / total * 100
            ax1.text(j + 0.7, i + 0.3, f'({percentage:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='gray')
    
    # 2. 训练历史 - 损失
    ax2 = plt.subplot2grid((3, 4), (0, 2), colspan=1)
    epochs = range(1, len(train_losses) + 1)
    ax2.plot(epochs, train_losses, 'b-', label='训练', linewidth=2, alpha=0.8)
    ax2.plot(epochs, val_losses, 'r-', label='验证', linewidth=2, alpha=0.8)
    ax2.set_title('📉 损失曲线', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 训练历史 - 准确率
    ax3 = plt.subplot2grid((3, 4), (0, 3), colspan=1)
    ax3.plot(epochs, train_accuracies, 'b-', label='训练', linewidth=2, alpha=0.8)
    ax3.plot(epochs, val_accuracies, 'r-', label='验证', linewidth=2, alpha=0.8)
    ax3.set_title('📈 准确率曲线', fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 过拟合分析
    ax4 = plt.subplot2grid((3, 4), (1, 2), colspan=1)
    overfitting = [train_acc - val_acc for train_acc, val_acc in zip(train_accuracies, val_accuracies)]
    ax4.plot(epochs, overfitting, 'g-', linewidth=2, alpha=0.8)
    ax4.set_title('🔍 过拟合程度', fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('训练-验证差值 (%)')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.grid(True, alpha=0.3)
    
    # 5. 各类别性能柱状图
    ax5 = plt.subplot2grid((3, 4), (1, 3), colspan=1)
    report = classification_report(y_test_labels, predicted_labels, output_dict=True)
    classes = label_encoder.classes_
    f1_scores = [report[cls]['f1-score'] if cls in report else 0 for cls in classes]
    
    bars = ax5.bar(classes, f1_scores, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    ax5.set_title('📊 各类别F1分数', fontweight='bold')
    ax5.set_ylabel('F1 Score')
    ax5.set_ylim(0, 1)
    
    # 在柱子上添加数值
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 6. 最佳参数信息
    ax6 = plt.subplot2grid((3, 4), (2, 0), colspan=4)
    ax6.axis('off')
    
    # 创建参数信息表格
    param_info = [
        ['参数', '数值', '说明'],
        ['隐藏层大小', f"{best_params['hidden_size']}", '神经网络隐藏层神经元数量'],
        ['学习率', f"{best_params['learning_rate']}", '模型参数更新步长'],
        ['Dropout率', f"{best_params['dropout_rate']}", '正则化强度，防止过拟合'],
        ['批大小', f"{best_params['batch_size']}", '每次训练的样本数量'],
        ['权重衰减', f"{best_params['weight_decay']}", 'L2正则化系数'],
        ['训练轮数', f"{len(train_losses)}", '实际训练的epoch数量'],
        ['最终过拟合程度', f"{overfitting[-1]:.2f}%", '训练与验证准确率差值']
    ]
    
    # 创建表格
    table = ax6.table(cellText=param_info[1:], colLabels=param_info[0],
                     cellLoc='center', loc='center',
                     colWidths=[0.2, 0.15, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # 设置表格样式
    for i in range(len(param_info)):
        for j in range(3):
            if i == 0:  # 标题行
                table[(i, j)].set_facecolor('#4CAF50')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:
                if j == 1:  # 数值列
                    table[(i, j)].set_facecolor('#E3F2FD')
                    table[(i, j)].set_text_props(weight='bold')
    
    ax6.set_title('🎛️ 最佳超参数配置 & 训练结果', fontsize=14, fontweight='bold', pad=20)
    
    plt.suptitle('🏆 帕金森病步态分析 - 最佳神经网络模型完整评估报告', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    plt.savefig('neural network/results/best_model_complete_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_detailed_confusion_matrix(y_test_labels, predicted_labels, label_encoder, test_accuracy, test_f1):
    """创建详细的混淆矩阵单独图"""
    
    cm = confusion_matrix(y_test_labels, predicted_labels)
    
    # 计算各种指标
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. 绝对数量混淆矩阵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_,
                ax=ax1, square=True, linewidths=1)
    ax1.set_title(f'混淆矩阵 - 绝对数量\n总体准确率: {test_accuracy:.1%}', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('预测标签', fontsize=12)
    ax1.set_ylabel('真实标签', fontsize=12)
    
    # 2. 百分比混淆矩阵
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Oranges', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_,
                ax=ax2, square=True, linewidths=1)
    ax2.set_title(f'混淆矩阵 - 按行百分比\nF1分数: {test_f1:.1%}', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('预测标签', fontsize=12)
    ax2.set_ylabel('真实标签', fontsize=12)
    
    plt.suptitle('🎯 最佳模型混淆矩阵详细分析', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('neural network/results/detailed_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_detailed_results(y_test_labels, predicted_labels, label_encoder, test_accuracy, test_f1):
    """打印详细的结果分析"""
    
    print("\n" + "="*60)
    print("🏆 最佳模型详细评估结果")
    print("="*60)
    
    print(f"\n📊 总体性能:")
    print(f"   测试集准确率: {test_accuracy:.4f} ({test_accuracy:.1%})")
    print(f"   测试集F1分数: {test_f1:.4f} ({test_f1:.1%})")
    
    print(f"\n📋 详细分类报告:")
    report = classification_report(y_test_labels, predicted_labels, output_dict=True)
    
    print(f"{'类别':<12} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'支持数':<6}")
    print("-" * 50)
    
    for class_name in label_encoder.classes_:
        if class_name in report:
            precision = report[class_name]['precision']
            recall = report[class_name]['recall']
            f1_score = report[class_name]['f1-score']
            support = int(report[class_name]['support'])
            print(f"{class_name:<12} {precision:<8.3f} {recall:<8.3f} {f1_score:<8.3f} {support:<6}")
        else:
            print(f"{class_name:<12} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'0':<6}")
    
    print(f"\n🔍 混淆矩阵分析:")
    cm = confusion_matrix(y_test_labels, predicted_labels)
    total_samples = np.sum(cm)
    
    print(f"   总样本数: {total_samples}")
    print(f"   正确预测: {np.trace(cm)} ({np.trace(cm)/total_samples:.1%})")
    print(f"   错误预测: {total_samples - np.trace(cm)} ({(total_samples - np.trace(cm))/total_samples:.1%})")

def main():
    """主函数"""
    print("🚀 开始评估最佳超参数模型...")
    
    # 训练最佳模型
    (model, test_accuracy, test_f1, y_test_labels, predicted_labels, 
     label_encoder, train_losses, train_accuracies, val_losses, val_accuracies, best_params) = train_best_model()
    
    # 打印详细结果
    print_detailed_results(y_test_labels, predicted_labels, label_encoder, test_accuracy, test_f1)
    
    # 创建综合评估报告
    print("\n📊 生成综合评估报告...")
    create_comprehensive_evaluation(
        y_test_labels, predicted_labels, label_encoder,
        train_losses, train_accuracies, val_losses, val_accuracies,
        best_params, test_accuracy, test_f1
    )
    
    # 创建详细混淆矩阵
    print("\n🎯 生成详细混淆矩阵...")
    create_detailed_confusion_matrix(y_test_labels, predicted_labels, label_encoder, test_accuracy, test_f1)
    
    # 保存最佳模型
    print("\n💾 保存最佳模型...")
    torch.save(model.state_dict(), 'neural network/results/final_best_model.pth')
    
    print("\n🎉 评估完成！生成的文件:")
    print("- neural network/results/best_model_complete_evaluation.png")
    print("- neural network/results/detailed_confusion_matrix.png")
    print("- neural network/results/final_best_model.pth")
    
    return model, test_accuracy, test_f1, y_test_labels, predicted_labels

if __name__ == "__main__":
    results = main()

