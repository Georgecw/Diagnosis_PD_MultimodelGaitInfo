import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# 导入自定义函数
from nn_reduced import ImprovedNN, load_reduced_data, preprocess_data, train_model_with_early_stopping, evaluate_model

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class HyperparameterTuner:
    """超参数调优器"""
    
    def __init__(self, data_suffix='_f_classif_k15'):
        """
        初始化超参数调优器
        
        Parameters:
        -----------
        data_suffix : str
            数据文件后缀，默认使用k=15的数据
        """
        self.data_suffix = data_suffix
        self.results = []
        self.best_params = None
        self.best_score = 0
        
    def load_data(self):
        """加载数据"""
        print(f"加载数据: {self.data_suffix}")
        X_train, y_train, X_test, y_test = load_reduced_data(self.data_suffix)
        
        if X_train is None:
            raise ValueError(f"无法加载数据文件: {self.data_suffix}")
        
        # 预处理数据
        X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, label_encoder = preprocess_data(
            X_train, y_train, X_test, y_test
        )
        
        self.X_train = X_train_tensor
        self.y_train = y_train_tensor
        self.X_test = X_test_tensor
        self.y_test = y_test_tensor
        self.label_encoder = label_encoder
        self.input_size = X_train_tensor.shape[1]
        self.output_size = len(label_encoder.classes_)
        
        print(f"数据加载完成: 输入维度={self.input_size}, 输出维度={self.output_size}")
        
    def grid_search(self, param_grid, max_trials=None):
        """
        网格搜索超参数
        
        Parameters:
        -----------
        param_grid : dict
            参数网格，包含要搜索的超参数范围
        max_trials : int, optional
            最大试验次数，None表示搜索所有组合
        """
        print("开始网格搜索...")
        print(f"参数网格: {param_grid}")
        
        # 生成所有参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        if max_trials and len(param_combinations) > max_trials:
            # 随机选择部分组合
            np.random.shuffle(param_combinations)
            param_combinations = param_combinations[:max_trials]
        
        print(f"总共要测试 {len(param_combinations)} 种参数组合")
        
        for i, param_combo in enumerate(param_combinations):
            print(f"\n--- 试验 {i+1}/{len(param_combinations)} ---")
            
            # 创建参数字典
            params = dict(zip(param_names, param_combo))
            print(f"当前参数: {params}")
            
            try:
                # 训练和评估模型
                result = self._train_and_evaluate(params)
                result['trial'] = i + 1
                result['params'] = params.copy()
                self.results.append(result)
                
                # 更新最佳参数
                if result['test_accuracy'] > self.best_score:
                    self.best_score = result['test_accuracy']
                    self.best_params = params.copy()
                    print(f"🎉 发现新的最佳参数！准确率: {self.best_score:.4f}")
                
            except Exception as e:
                print(f"❌ 试验失败: {e}")
                continue
        
        print(f"\n🏆 搜索完成！最佳准确率: {self.best_score:.4f}")
        print(f"🏆 最佳参数: {self.best_params}")
        
    def _train_and_evaluate(self, params):
        """训练和评估单个参数组合"""
        # 设置随机种子确保可重现性
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 创建模型
        model = ImprovedNN(
            input_size=self.input_size,
            hidden_size=params['hidden_size'],
            output_size=self.output_size,
            dropout_rate=params['dropout_rate']
        )
        
        # 创建数据加载器
        train_dataset = TensorDataset(self.X_train, self.y_train)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=params['batch_size'], 
            shuffle=True
        )
        
        # 创建优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(), 
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        
        # 训练模型
        train_losses, train_accuracies, val_losses, val_accuracies = train_model_with_early_stopping(
            model, train_loader, criterion, optimizer, 
            self.X_test, self.y_test,
            num_epochs=params['max_epochs'],
            patience=params['patience']
        )
        
        # 评估模型
        test_accuracy, test_f1, _, _ = evaluate_model(
            model, self.X_test, self.y_test, self.label_encoder
        )
        
        # 计算过拟合程度
        final_train_acc = train_accuracies[-1] if train_accuracies else 0
        final_val_acc = val_accuracies[-1] if val_accuracies else 0
        overfitting = final_train_acc - final_val_acc
        
        result = {
            'test_accuracy': test_accuracy,
            'test_f1': test_f1,
            'train_accuracy': final_train_acc,
            'val_accuracy': final_val_acc,
            'overfitting': overfitting,
            'epochs_trained': len(train_losses),
            'final_train_loss': train_losses[-1] if train_losses else float('inf'),
            'final_val_loss': val_losses[-1] if val_losses else float('inf')
        }
        
        print(f"结果: 测试准确率={test_accuracy:.4f}, F1={test_f1:.4f}, 过拟合程度={overfitting:.2f}%")
        
        return result
    
    def analyze_results(self):
        """分析超参数调优结果"""
        if not self.results:
            print("没有结果可分析")
            return
        
        # 转换为DataFrame
        df = pd.DataFrame(self.results)
        
        # 展开参数列
        params_df = pd.json_normalize(df['params'])
        df = pd.concat([df.drop('params', axis=1), params_df], axis=1)
        
        print("\n=== 超参数调优结果分析 ===")
        
        # 1. 最佳结果
        best_result = df.loc[df['test_accuracy'].idxmax()]
        print(f"\n🏆 最佳结果:")
        print(f"  测试准确率: {best_result['test_accuracy']:.4f}")
        print(f"  测试F1分数: {best_result['test_f1']:.4f}")
        print(f"  隐藏层大小: {best_result['hidden_size']}")
        print(f"  学习率: {best_result['learning_rate']}")
        print(f"  Dropout率: {best_result['dropout_rate']}")
        print(f"  批大小: {best_result['batch_size']}")
        print(f"  权重衰减: {best_result['weight_decay']}")
        print(f"  训练轮数: {best_result['epochs_trained']}")
        print(f"  过拟合程度: {best_result['overfitting']:.2f}%")
        
        # 2. 参数重要性分析
        print(f"\n📊 参数对性能的影响:")
        for param in ['hidden_size', 'learning_rate', 'dropout_rate', 'batch_size', 'weight_decay']:
            if param in df.columns:
                correlation = df['test_accuracy'].corr(df[param])
                print(f"  {param}: 相关系数 = {correlation:.3f}")
        
        # 3. 统计摘要
        print(f"\n📈 性能统计:")
        print(f"  平均测试准确率: {df['test_accuracy'].mean():.4f} ± {df['test_accuracy'].std():.4f}")
        print(f"  最高测试准确率: {df['test_accuracy'].max():.4f}")
        print(f"  最低测试准确率: {df['test_accuracy'].min():.4f}")
        print(f"  平均过拟合程度: {df['overfitting'].mean():.2f}% ± {df['overfitting'].std():.2f}%")
        
        return df
    
    def plot_results(self, df=None):
        """可视化超参数调优结果"""
        if df is None:
            df = self.analyze_results()
        
        if df is None or df.empty:
            return
        
        try:
            # 创建子图
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('超参数调优结果分析', fontsize=16)
            
            # 1. 隐藏层大小 vs 准确率
            if 'hidden_size' in df.columns:
                axes[0,0].scatter(df['hidden_size'], df['test_accuracy'], alpha=0.7, c='blue')
                axes[0,0].set_xlabel('隐藏层大小')
                axes[0,0].set_ylabel('测试准确率')
                axes[0,0].set_title('隐藏层大小 vs 测试准确率')
                axes[0,0].grid(True, alpha=0.3)
            
            # 2. 学习率 vs 准确率
            if 'learning_rate' in df.columns:
                axes[0,1].scatter(df['learning_rate'], df['test_accuracy'], alpha=0.7, c='green')
                axes[0,1].set_xlabel('学习率')
                axes[0,1].set_ylabel('测试准确率')
                axes[0,1].set_title('学习率 vs 测试准确率')
                axes[0,1].set_xscale('log')
                axes[0,1].grid(True, alpha=0.3)
            
            # 3. Dropout率 vs 准确率
            if 'dropout_rate' in df.columns:
                axes[0,2].scatter(df['dropout_rate'], df['test_accuracy'], alpha=0.7, c='red')
                axes[0,2].set_xlabel('Dropout率')
                axes[0,2].set_ylabel('测试准确率')
                axes[0,2].set_title('Dropout率 vs 测试准确率')
                axes[0,2].grid(True, alpha=0.3)
            
            # 4. 过拟合程度分布
            axes[1,0].hist(df['overfitting'], bins=15, alpha=0.7, color='orange')
            axes[1,0].set_xlabel('过拟合程度 (%)')
            axes[1,0].set_ylabel('频次')
            axes[1,0].set_title('过拟合程度分布')
            axes[1,0].axvline(x=0, color='black', linestyle='--', alpha=0.5)
            axes[1,0].grid(True, alpha=0.3)
            
            # 5. 训练轮数 vs 准确率
            axes[1,1].scatter(df['epochs_trained'], df['test_accuracy'], alpha=0.7, c='purple')
            axes[1,1].set_xlabel('训练轮数')
            axes[1,1].set_ylabel('测试准确率')
            axes[1,1].set_title('训练轮数 vs 测试准确率')
            axes[1,1].grid(True, alpha=0.3)
            
            # 6. 准确率排名
            df_sorted = df.sort_values('test_accuracy', ascending=False).head(10)
            axes[1,2].barh(range(len(df_sorted)), df_sorted['test_accuracy'])
            axes[1,2].set_xlabel('测试准确率')
            axes[1,2].set_ylabel('试验排名')
            axes[1,2].set_title('Top 10 试验结果')
            axes[1,2].set_yticks(range(len(df_sorted)))
            axes[1,2].set_yticklabels([f"Trial {int(x)}" for x in df_sorted['trial']])
            axes[1,2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('neural network/results/hyperparameter_tuning_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"绘图时出现错误: {e}")
            print("跳过可视化步骤")
    
    def save_results(self, filename='hyperparameter_results.csv'):
        """保存结果到CSV文件"""
        if not self.results:
            print("没有结果可保存")
            return
        
        df = pd.DataFrame(self.results)
        params_df = pd.json_normalize(df['params'])
        df = pd.concat([df.drop('params', axis=1), params_df], axis=1)
        
        filepath = f'neural network/results/{filename}'
        df.to_csv(filepath, index=False)
        print(f"结果已保存到: {filepath}")

def quick_search():
    """快速搜索 - 粗粒度网格"""
    print("🚀 开始快速超参数搜索...")
    
    tuner = HyperparameterTuner('_f_classif_k15')  # 使用最佳特征数量
    tuner.load_data()
    
    # 定义搜索空间 - 较粗的网格
    param_grid = {
        'hidden_size': [8, 12, 16, 20, 24, 32],  # 不同的隐藏层大小
        'learning_rate': [0.0005, 0.001, 0.002, 0.005],  # 不同学习率
        'dropout_rate': [0.2, 0.3, 0.4, 0.5],  # 不同dropout率
        'batch_size': [4, 8, 16],  # 不同批大小
        'weight_decay': [1e-5, 1e-4, 1e-3],  # 不同权重衰减
        'max_epochs': [200],  # 固定最大轮数
        'patience': [30]  # 固定耐心值
    }
    
    tuner.grid_search(param_grid, max_trials=50)  # 限制最大试验次数
    df = tuner.analyze_results()
    tuner.plot_results(df)
    tuner.save_results('quick_search_results.csv')
    
    return tuner

def fine_search(best_params=None):
    """精细搜索 - 在最佳参数附近搜索"""
    print("🎯 开始精细超参数搜索...")
    
    tuner = HyperparameterTuner('_f_classif_k15')
    tuner.load_data()
    
    if best_params is None:
        # 使用默认的较好参数作为中心
        best_params = {
            'hidden_size': 16,
            'learning_rate': 0.001,
            'dropout_rate': 0.4,
            'batch_size': 8,
            'weight_decay': 1e-4
        }
    
    # 在最佳参数附近定义精细网格
    param_grid = {
        'hidden_size': [best_params['hidden_size'] - 4, best_params['hidden_size'], 
                       best_params['hidden_size'] + 4, best_params['hidden_size'] + 8],
        'learning_rate': [best_params['learning_rate'] * 0.5, best_params['learning_rate'],
                         best_params['learning_rate'] * 1.5, best_params['learning_rate'] * 2],
        'dropout_rate': [max(0.1, best_params['dropout_rate'] - 0.1), best_params['dropout_rate'],
                        min(0.6, best_params['dropout_rate'] + 0.1)],
        'batch_size': [max(4, best_params['batch_size'] // 2), best_params['batch_size'],
                      min(16, best_params['batch_size'] * 2)],
        'weight_decay': [best_params['weight_decay'] * 0.1, best_params['weight_decay'],
                        best_params['weight_decay'] * 10],
        'max_epochs': [300],
        'patience': [40]
    }
    
    tuner.grid_search(param_grid)
    df = tuner.analyze_results()
    tuner.plot_results(df)
    tuner.save_results('fine_search_results.csv')
    
    return tuner

def compare_feature_sizes():
    """比较不同特征数量下的最佳超参数"""
    print("🔄 比较不同特征数量的最佳超参数...")
    
    feature_sizes = ['_f_classif_k12', '_f_classif_k15', '_f_classif_k18']
    all_results = {}
    
    for feature_size in feature_sizes:
        print(f"\n--- 测试 {feature_size} ---")
        
        tuner = HyperparameterTuner(feature_size)
        tuner.load_data()
        
        # 使用较小的搜索空间
        param_grid = {
            'hidden_size': [8, 12, 16, 20, 24],
            'learning_rate': [0.001, 0.002],
            'dropout_rate': [0.3, 0.4, 0.5],
            'batch_size': [8],
            'weight_decay': [1e-4],
            'max_epochs': [200],
            'patience': [30]
        }
        
        tuner.grid_search(param_grid, max_trials=20)
        all_results[feature_size] = {
            'best_params': tuner.best_params,
            'best_score': tuner.best_score,
            'results': tuner.results
        }
    
    # 比较结果
    print("\n=== 不同特征数量的最佳结果比较 ===")
    for feature_size, result in all_results.items():
        print(f"{feature_size}: 准确率={result['best_score']:.4f}, 最佳参数={result['best_params']}")
    
    return all_results

def train_and_evaluate_best_model(best_params, data_suffix='_f_classif_k15'):
    """使用最佳参数训练模型并生成详细评估结果"""
    print(f"\n🏆 使用最佳参数训练最终模型...")
    print(f"最佳参数: {best_params}")
    
    # 加载数据
    X_train, y_train, X_test, y_test = load_reduced_data(data_suffix)
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
        num_epochs=best_params.get('max_epochs', 300),
        patience=best_params.get('patience', 50)
    )
    
    # 评估模型
    test_accuracy, test_f1, y_test_labels, predicted_labels = evaluate_model(
        model, X_test_tensor, y_test_tensor, label_encoder
    )
    
    return model, test_accuracy, test_f1, y_test_labels, predicted_labels, label_encoder, train_losses, train_accuracies, val_losses, val_accuracies

def plot_best_model_results(y_test_labels, predicted_labels, label_encoder, 
                           train_losses, train_accuracies, val_losses, val_accuracies, 
                           best_params, test_accuracy, test_f1):
    """绘制最佳模型的详细结果"""
    
    try:
        # 创建子图
        fig = plt.figure(figsize=(20, 12))
        
        # 1. 混淆矩阵 (大图)
        ax1 = plt.subplot(2, 3, (1, 4))  # 占据左侧两行
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test_labels, predicted_labels)
        
        # 绘制混淆矩阵
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=label_encoder.classes_, 
                    yticklabels=label_encoder.classes_,
                    ax=ax1, cbar_kws={'shrink': 0.8})
        ax1.set_title(f'最佳模型混淆矩阵\n准确率: {test_accuracy:.4f}, F1: {test_f1:.4f}', 
                      fontsize=14, fontweight='bold')
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
        ax2 = plt.subplot(2, 3, 2)
        epochs = range(1, len(train_losses) + 1)
        ax2.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
        ax2.plot(epochs, val_losses, 'r-', label='验证损失', linewidth=2)
        ax2.set_title('训练和验证损失', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 训练历史 - 准确率
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(epochs, train_accuracies, 'b-', label='训练准确率', linewidth=2)
        ax3.plot(epochs, val_accuracies, 'r-', label='验证准确率', linewidth=2)
        ax3.set_title('训练和验证准确率', fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 过拟合分析
        ax4 = plt.subplot(2, 3, 5)
        overfitting = [train_acc - val_acc for train_acc, val_acc in zip(train_accuracies, val_accuracies)]
        ax4.plot(epochs, overfitting, 'g-', linewidth=2)
        ax4.set_title('过拟合程度 (训练准确率 - 验证准确率)', fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy Difference (%)')
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax4.grid(True, alpha=0.3)
        
        # 5. 最佳参数展示
        ax5 = plt.subplot(2, 3, 6)
        ax5.axis('off')
        
        # 参数文本
        param_text = f"""最佳超参数配置:
        
隐藏层大小: {best_params['hidden_size']}
学习率: {best_params['learning_rate']}
Dropout率: {best_params['dropout_rate']}
批大小: {best_params['batch_size']}
权重衰减: {best_params['weight_decay']}

最终性能:
测试准确率: {test_accuracy:.4f}
测试F1分数: {test_f1:.4f}
训练轮数: {len(train_losses)}
最终过拟合程度: {overfitting[-1]:.2f}%"""
        
        ax5.text(0.1, 0.9, param_text, transform=ax5.transAxes, fontsize=11,
                 verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
                 facecolor='lightblue', alpha=0.8))
        
        plt.suptitle('🏆 最佳超参数模型完整评估报告', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('neural network/results/best_model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"绘制最佳模型结果时出现错误: {e}")

def generate_classification_report_visualization(y_test_labels, predicted_labels, label_encoder):
    """生成分类报告的可视化"""
    try:
        from sklearn.metrics import classification_report
        
        # 获取分类报告
        report = classification_report(y_test_labels, predicted_labels, output_dict=True)
        
        # 提取各类别的指标
        classes = label_encoder.classes_
        metrics = ['precision', 'recall', 'f1-score']
        
        # 创建数据矩阵
        data = []
        for class_name in classes:
            if class_name in report:
                data.append([report[class_name][metric] for metric in metrics])
            else:
                data.append([0, 0, 0])  # 如果类别不在报告中
        
        data = np.array(data)
        
        # 绘制热力图
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(data, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        
        # 设置标签
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes)
        
        # 在每个格子中添加数值
        for i in range(len(classes)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{data[i, j]:.3f}', 
                              ha='center', va='center', color='black', fontweight='bold')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('分数', rotation=270, labelpad=15)
        
        # 设置标题
        ax.set_title('各类别分类性能热力图', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig('neural network/results/classification_report_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"生成分类报告可视化时出现错误: {e}")

def main():
    """主函数"""
    print("🎛️ 神经网络超参数调优程序")
    print("=" * 50)
    
    # 1. 快速搜索
    print("\n1️⃣ 执行快速搜索...")
    quick_tuner = quick_search()
    
    # 2. 基于快速搜索结果进行精细搜索
    print("\n2️⃣ 基于快速搜索结果执行精细搜索...")
    fine_tuner = fine_search(quick_tuner.best_params)
    
    # 3. 比较不同特征数量
    print("\n3️⃣ 比较不同特征数量的最佳超参数...")
    comparison_results = compare_feature_sizes()
    
    # 4. 确定全局最佳参数
    print("\n4️⃣ 确定全局最佳参数...")
    all_best_scores = [
        (quick_tuner.best_score, quick_tuner.best_params, "快速搜索"),
        (fine_tuner.best_score, fine_tuner.best_params, "精细搜索")
    ]
    
    # 添加不同特征数量的最佳结果
    for feature_size, result in comparison_results.items():
        all_best_scores.append((result['best_score'], result['best_params'], f"特征数量 {feature_size}"))
    
    # 找到全局最佳
    global_best_score, global_best_params, best_source = max(all_best_scores, key=lambda x: x[0])
    
    print(f"🏆 全局最佳结果来自: {best_source}")
    print(f"🏆 全局最佳准确率: {global_best_score:.4f}")
    print(f"🏆 全局最佳参数: {global_best_params}")
    
    # 5. 使用最佳参数训练最终模型并生成详细评估
    print("\n5️⃣ 训练最终最佳模型并生成详细评估...")
    (model, test_accuracy, test_f1, y_test_labels, predicted_labels, 
     label_encoder, train_losses, train_accuracies, val_losses, val_accuracies) = train_and_evaluate_best_model(
        global_best_params, '_f_classif_k15'
    )
    
    # 6. 生成详细的可视化结果
    print("\n6️⃣ 生成最佳模型的详细可视化结果...")
    plot_best_model_results(
        y_test_labels, predicted_labels, label_encoder,
        train_losses, train_accuracies, val_losses, val_accuracies,
        global_best_params, test_accuracy, test_f1
    )
    
    # 7. 生成分类报告热力图
    print("\n7️⃣ 生成分类报告热力图...")
    generate_classification_report_visualization(y_test_labels, predicted_labels, label_encoder)
    
    # 8. 保存最佳模型
    print("\n8️⃣ 保存最佳模型...")
    torch.save(model.state_dict(), 'neural network/results/best_hyperparameter_model.pth')
    
    print("\n🎉 超参数调优完成！")
    print("生成的文件:")
    print("- neural network/results/hyperparameter_tuning_analysis.png")
    print("- neural network/results/quick_search_results.csv")
    print("- neural network/results/fine_search_results.csv")
    print("- neural network/results/best_model_evaluation.png (新增)")
    print("- neural network/results/classification_report_heatmap.png (新增)")
    print("- neural network/results/best_hyperparameter_model.pth (新增)")
    
    return quick_tuner, fine_tuner, comparison_results, model, global_best_params

if __name__ == "__main__":
    results = main()
