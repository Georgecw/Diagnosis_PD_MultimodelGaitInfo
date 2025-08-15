import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

class FeatureSelector:
    """基于统计检验的特征选择器"""
    
    def __init__(self, method='f_classif', k=15):
        """
        初始化特征选择器
        
        Parameters:
        -----------
        method : str
            特征选择方法 ('f_classif', 'chi2', 'mutual_info')
        k : int
            选择的特征数量
        """
        self.method = method
        self.k = k
        self.selector = None
        self.feature_scores = None
        self.selected_features = None
        self.feature_rankings = None
        
    def fit_transform(self, X_train, y_train, X_test=None):
        """
        拟合特征选择器并转换数据
        
        Parameters:
        -----------
        X_train : DataFrame
            训练集特征
        y_train : Series or array
            训练集标签
        X_test : DataFrame, optional
            测试集特征
            
        Returns:
        --------
        X_train_selected : DataFrame
            选择后的训练集特征
        X_test_selected : DataFrame or None
            选择后的测试集特征
        """
        print(f"使用 {self.method} 方法进行特征选择...")
        print(f"原始特征数量: {X_train.shape[1]}")
        print(f"目标选择特征数量: {self.k}")
        
        # 选择评分函数
        if self.method == 'f_classif':
            score_func = f_classif
        elif self.method == 'chi2':
            score_func = chi2
            # Chi2需要非负特征，进行标准化
            scaler = MinMaxScaler()
            X_train = pd.DataFrame(
                scaler.fit_transform(X_train), 
                columns=X_train.columns, 
                index=X_train.index
            )
            if X_test is not None:
                X_test = pd.DataFrame(
                    scaler.transform(X_test), 
                    columns=X_test.columns, 
                    index=X_test.index
                )
        elif self.method == 'mutual_info':
            score_func = mutual_info_classif
        else:
            raise ValueError("method must be one of: 'f_classif', 'chi2', 'mutual_info'")
        
        # 创建选择器
        self.selector = SelectKBest(score_func=score_func, k=self.k)
        
        # 拟合并转换训练数据
        X_train_selected = self.selector.fit_transform(X_train, y_train)
        
        # 获取特征分数和选择的特征
        self.feature_scores = self.selector.scores_
        selected_mask = self.selector.get_support()
        self.selected_features = X_train.columns[selected_mask].tolist()
        
        # 创建特征排名
        feature_score_pairs = list(zip(X_train.columns, self.feature_scores))
        feature_score_pairs.sort(key=lambda x: x[1], reverse=True)
        self.feature_rankings = feature_score_pairs
        
        # 转换为DataFrame
        X_train_selected = pd.DataFrame(
            X_train_selected, 
            columns=self.selected_features,
            index=X_train.index
        )
        
        X_test_selected = None
        if X_test is not None:
            X_test_selected = self.selector.transform(X_test)
            X_test_selected = pd.DataFrame(
                X_test_selected, 
                columns=self.selected_features,
                index=X_test.index
            )
        
        print(f"实际选择特征数量: {len(self.selected_features)}")
        print("选择的特征:", self.selected_features)
        
        return X_train_selected, X_test_selected
    
    def plot_feature_scores(self, top_n=20, figsize=(12, 8)):
        """绘制特征重要性分数"""
        if self.feature_rankings is None:
            print("请先运行 fit_transform 方法")
            return
        
        # 取前top_n个特征
        top_features = self.feature_rankings[:top_n]
        features, scores = zip(*top_features)
        
        plt.figure(figsize=figsize)
        colors = ['red' if feat in self.selected_features else 'lightblue' for feat in features]
        
        bars = plt.barh(range(len(features)), scores, color=colors)
        plt.yticks(range(len(features)), features)
        plt.xlabel(f'{self.method} 分数')
        plt.title(f'特征重要性排名 (前{top_n}个特征)')
        plt.gca().invert_yaxis()
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label=f'选择的特征 (前{self.k}个)'),
            Patch(facecolor='lightblue', label='未选择的特征')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(f'neural network/results/feature_importance_{self.method}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_groups_analysis(self):
        """分析不同特征组的选择情况"""
        if self.selected_features is None:
            print("请先运行 fit_transform 方法")
            return
        
        # 按特征组分类
        feature_groups = {
            'gait_features': [],
            'spectral_features': [], 
            'temporal_features': []
        }
        
        for feature in self.selected_features:
            if 'gait_features' in feature:
                feature_groups['gait_features'].append(feature)
            elif 'spectral_features' in feature:
                feature_groups['spectral_features'].append(feature)
            elif 'temporal_features' in feature:
                feature_groups['temporal_features'].append(feature)
        
        # 统计各组选择的特征数量
        group_counts = {group: len(features) for group, features in feature_groups.items()}
        
        # 原始各组特征数量（从特征名推断）
        all_features = [feat for feat_list in self.feature_rankings for feat in [feat_list[0]]]
        original_counts = {
            'gait_features': len([f for f in all_features if 'gait_features' in f]),
            'spectral_features': len([f for f in all_features if 'spectral_features' in f]),
            'temporal_features': len([f for f in all_features if 'temporal_features' in f])
        }
        
        # 绘制对比图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 原始特征分布
        groups = list(original_counts.keys())
        original_values = list(original_counts.values())
        selected_values = [group_counts.get(group, 0) for group in groups]
        
        x = np.arange(len(groups))
        width = 0.35
        
        ax1.bar(x - width/2, original_values, width, label='原始特征数', alpha=0.8)
        ax1.bar(x + width/2, selected_values, width, label='选择特征数', alpha=0.8)
        ax1.set_xlabel('特征组')
        ax1.set_ylabel('特征数量')
        ax1.set_title('各特征组的特征选择情况')
        ax1.set_xticks(x)
        ax1.set_xticklabels([g.replace('_', '\n') for g in groups])
        ax1.legend()
        
        # 选择比例
        selection_ratios = [selected_values[i] / original_values[i] if original_values[i] > 0 else 0 
                          for i in range(len(groups))]
        
        ax2.bar(groups, selection_ratios, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax2.set_ylabel('选择比例')
        ax2.set_title('各特征组的选择比例')
        ax2.set_xticklabels([g.replace('_', '\n') for g in groups])
        
        for i, ratio in enumerate(selection_ratios):
            ax2.text(i, ratio + 0.01, f'{ratio:.2f}', ha='center')
        
        plt.tight_layout()
        plt.savefig('neural network/results/feature_groups_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印详细信息
        print("\n=== 特征组分析 ===")
        for group in groups:
            print(f"{group}:")
            print(f"  原始特征数: {original_counts[group]}")
            print(f"  选择特征数: {group_counts.get(group, 0)}")
            print(f"  选择比例: {selection_ratios[groups.index(group)]:.2%}")
            if group_counts.get(group, 0) > 0:
                print(f"  选择的特征: {feature_groups[group]}")
            print()

def load_and_preprocess_data():
    """加载和预处理数据"""
    print("正在加载数据...")
    
    # 加载数据
    X_train = pd.read_csv('neural network/X_train_reduced.csv')
    y_train = pd.read_csv('neural network/y_train.csv')
    X_test = pd.read_csv('neural network/X_test_reduced.csv')
    y_test = pd.read_csv('neural network/y_test.csv')
    
    # 移除空行
    X_train = X_train.dropna()
    y_train = y_train.dropna()
    X_test = X_test.dropna()
    y_test = y_test.dropna()
    
    print(f"训练数据形状: {X_train.shape}")
    print(f"测试数据形状: {X_test.shape}")
    print(f"类别分布:\n{y_train['Group'].value_counts()}")
    
    return X_train, y_train['Group'], X_test, y_test['Group']

def compare_feature_selection_methods(X_train, y_train, X_test, y_test, k_values=[10, 12, 15, 18]):
    """比较不同特征选择方法和不同k值的效果"""
    methods = ['f_classif', 'mutual_info']  # chi2需要非负特征，先排除
    results = []
    
    print("=== 比较不同特征选择方法 ===")
    
    for method in methods:
        for k in k_values:
            print(f"\n--- {method} with k={k} ---")
            
            selector = FeatureSelector(method=method, k=k)
            X_train_selected, X_test_selected = selector.fit_transform(X_train, y_train, X_test)
            
            # 记录结果
            results.append({
                'method': method,
                'k': k,
                'selected_features': selector.selected_features.copy(),
                'feature_scores': selector.feature_scores.copy(),
                'X_train_selected': X_train_selected.copy(),
                'X_test_selected': X_test_selected.copy()
            })
    
    return results

def save_selected_data(X_train_selected, y_train, X_test_selected, y_test, method, k):
    """保存选择后的数据"""
    print(f"\n保存 {method} k={k} 选择后的数据...")
    
    # 创建文件名
    suffix = f"_{method}_k{k}"
    
    # 保存训练数据
    X_train_selected.to_csv(f'neural network/X_train_selected{suffix}.csv', index=False)
    y_train.to_frame('Group').to_csv(f'neural network/y_train_selected{suffix}.csv', index=False)
    
    # 保存测试数据
    X_test_selected.to_csv(f'neural network/X_test_selected{suffix}.csv', index=False)
    y_test.to_frame('Group').to_csv(f'neural network/y_test_selected{suffix}.csv', index=False)
    
    print(f"数据已保存，文件后缀: {suffix}")

def main():
    """主函数"""
    # 加载数据
    X_train, y_train, X_test, y_test = load_and_preprocess_data()
    
    # 标签编码（用于统计检验）
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    print(f"\n类别编码映射:")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"  {class_name}: {i}")
    
    # 比较不同方法
    results = compare_feature_selection_methods(X_train, y_train_encoded, X_test, y_test_encoded)
    
    # 选择最佳方法（这里选择F-score，k=12作为推荐）
    print("\n=== 推荐方案: F-score, k=12 ===")
    best_selector = FeatureSelector(method='f_classif', k=12)
    X_train_best, X_test_best = best_selector.fit_transform(X_train, y_train_encoded, X_test)
    
    # 可视化分析
    print("\n生成特征重要性可视化...")
    best_selector.plot_feature_scores(top_n=29)  # 显示所有特征
    best_selector.plot_feature_groups_analysis()
    
    # 保存推荐方案的数据
    save_selected_data(X_train_best, y_train, X_test_best, y_test, 'f_classif', 12)
    
    # 额外保存k=15和k=18的版本供对比
    print("\n=== 额外方案: F-score, k=15 ===")
    selector_15 = FeatureSelector(method='f_classif', k=15)
    X_train_15, X_test_15 = selector_15.fit_transform(X_train, y_train_encoded, X_test)
    save_selected_data(X_train_15, y_train, X_test_15, y_test, 'f_classif', 15)
    
    print("\n=== 新增方案: F-score, k=18 ===")
    selector_18 = FeatureSelector(method='f_classif', k=18)
    X_train_18, X_test_18 = selector_18.fit_transform(X_train, y_train_encoded, X_test)
    save_selected_data(X_train_18, y_train, X_test_18, y_test, 'f_classif', 18)
    
    print("\n=== 特征选择完成 ===")
    print("生成的文件:")
    print("1. 特征重要性图: neural network/results/feature_importance_f_classif.png")
    print("2. 特征组分析图: neural network/results/feature_groups_analysis.png")
    print("3. 选择后的数据文件:")
    print("   - X_train_selected_f_classif_k12.csv")
    print("   - y_train_selected_f_classif_k12.csv") 
    print("   - X_test_selected_f_classif_k12.csv")
    print("   - y_test_selected_f_classif_k12.csv")
    print("   - X_train_selected_f_classif_k15.csv")
    print("   - X_train_selected_f_classif_k18.csv (新增)")
    print("   - 等...")
    
    return best_selector, X_train_best, X_test_best

if __name__ == "__main__":
    selector, X_train_selected, X_test_selected = main()
