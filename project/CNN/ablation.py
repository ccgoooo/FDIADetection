import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time
from detect_model import CNNTrainer, load_data  
from model import ResidualCNN, SimpleCNN ,PlainCNN7,ResidualCNN1 

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def reshape_data_if_needed(X_train, y_train, X_val, y_val, X_test, y_test, window_size=10):
    """如果数据是2D，则重塑为3D窗口格式"""
    if len(X_train.shape) == 2:
        print("检测到2D数据，进行窗口重塑...")
        # 计算可用的窗口数
        n_train = X_train.shape[0] // window_size
        n_val = X_val.shape[0] // window_size
        n_test = X_test.shape[0] // window_size
        
        X_train = X_train[:n_train * window_size].reshape(n_train, window_size, -1)
        X_val = X_val[:n_val * window_size].reshape(n_val, window_size, -1)
        X_test = X_test[:n_test * window_size].reshape(n_test, window_size, -1)
        
        # 每个窗口取最后一个时间步的标签（或第一个，需保持一致）
        y_train = y_train[:n_train * window_size:window_size]
        y_val = y_val[:n_val * window_size:window_size]
        y_test = y_test[:n_test * window_size:window_size]
        
        print(f"重塑后形状：X_train {X_train.shape}, X_val {X_val.shape}, X_test {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test

def evaluate_model(model, test_loader):
    """计算模型在测试集上的各类指标"""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(model.device)
            output = model.model(data)
            probs = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
            preds = output.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(target.numpy())
            all_probs.extend(probs)
    return np.array(all_preds), np.array(all_targets), np.array(all_probs)

def run_experiment_1():
    print("="*60)
    print("实验1：残差结构有效性验证")
    print("="*60)
    set_seed(42)

    # 加载数据
    X_train, y_train, X_val, y_val, X_test, y_test = load_data("./processed_data/uniform_mixed_84")
    X_train, y_train, X_val, y_val, X_test, y_test = reshape_data_if_needed(
        X_train, y_train, X_val, y_val, X_test, y_test, window_size=10
    )
    window_size = X_train.shape[1]
    feature_dim = X_train.shape[2]

    trainer = CNNTrainer()
    train_loader, val_loader, test_loader = trainer.prepare_data(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32
    )
    class_weights = trainer.compute_class_weights(y_train)

    results = []

    # 模型定义列表
    models = [
        ("A: PlainCNN7 (无残差)", PlainCNN7, "plain"),
        ("B: ResidualCNN-1 (1个残差块)", ResidualCNN1, "res1"),
        ("C: ResidualCNN-3 (3个残差块)", ResidualCNN, "res3")
    ]

    for name, ModelClass, tag in models:
        print(f"\n--- 训练 {name} ---")
        trainer.model = ModelClass(window_size=window_size, feature_dim=feature_dim, dropout_rate=0.3).to(trainer.device)
        print(f"参数量: {sum(p.numel() for p in trainer.model.parameters()):,}")

        start = time.time()
        history = trainer.train(train_loader, val_loader, epochs=50, learning_rate=0.001,
                                patience=10, class_weights=class_weights)
        train_time = time.time() - start

        preds, targets, probs = evaluate_model(trainer, test_loader)
        acc = accuracy_score(targets, preds)
        prec = precision_score(targets, preds)
        rec = recall_score(targets, preds)
        f1 = f1_score(targets, preds)
        auc = roc_auc_score(targets, probs)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "AUC": auc,
            "Train Time (s)": train_time
        })

    df = pd.DataFrame(results)
    print("\n\n实验1结果对比：")
    print(df.to_string(index=False))
    df.to_csv("ablation_experiment_1.csv", index=False)
    return df

def run_experiment_2(base_feature_dim=42):
    print("="*60)
    print("实验2：特征扩展有效性验证")
    print("="*60)
    
    set_seed(42)
    
    # 加载原始数据
    X_train, y_train, X_val, y_val, X_test, y_test = load_data("./processed_data")
    X_train, y_train, X_val, y_val, X_test, y_test = reshape_data_if_needed(
        X_train, y_train, X_val, y_val, X_test, y_test, window_size=10
    )
    
    window_size = X_train.shape[1]
    full_feature_dim = X_train.shape[2]
    
    # 构造基础特征数据集（取前 base_feature_dim 维）
    X_train_base = X_train[:, :, :base_feature_dim]
    X_val_base   = X_val[:, :, :base_feature_dim]
    X_test_base  = X_test[:, :, :base_feature_dim]
    
    print(f"基础特征维度: {base_feature_dim}, 扩展特征维度: {full_feature_dim}")
    
    trainer = CNNTrainer()
    results = []
    
    # 基础特征集
    print("\n--- 使用基础特征集 ({} 维) ---".format(base_feature_dim))
    train_loader_base, val_loader_base, test_loader_base = trainer.prepare_data(
        X_train_base, y_train, X_val_base, y_val, X_test_base, y_test, batch_size=32
    )
    class_weights = trainer.compute_class_weights(y_train)  # 权重基于原始标签计算，不变
    
    trainer.model = ResidualCNN(window_size=window_size, feature_dim=base_feature_dim, dropout_rate=0.3).to(trainer.device)
    print(f"参数量: {sum(p.numel() for p in trainer.model.parameters()):,}")
    
    start = time.time()
    history = trainer.train(train_loader_base, val_loader_base, epochs=50, learning_rate=0.001,
                            patience=10, class_weights=class_weights)
    train_time = time.time() - start
    
    preds, targets, probs = evaluate_model(trainer, test_loader_base)
    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds)
    rec = recall_score(targets, preds)
    f1 = f1_score(targets, preds)
    auc = roc_auc_score(targets, probs)
    
    results.append({
        "Dataset": f"基础特征 ({base_feature_dim}维)",
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "AUC": auc,
        "Train Time (s)": train_time
    })
    
    # 扩展特征集（全部特征）
    print("\n--- 使用扩展特征集 ({} 维) ---".format(full_feature_dim))
    train_loader_full, val_loader_full, test_loader_full = trainer.prepare_data(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32
    )
    # 重新创建模型（特征维度变化）
    trainer.model = ResidualCNN(window_size=window_size, feature_dim=full_feature_dim, dropout_rate=0.3).to(trainer.device)
    print(f"参数量: {sum(p.numel() for p in trainer.model.parameters()):,}")
    
    start = time.time()
    history = trainer.train(train_loader_full, val_loader_full, epochs=50, learning_rate=0.001,
                            patience=10, class_weights=class_weights)
    train_time = time.time() - start
    
    preds, targets, probs = evaluate_model(trainer, test_loader_full)
    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds)
    rec = recall_score(targets, preds)
    f1 = f1_score(targets, preds)
    auc = roc_auc_score(targets, probs)
    
    results.append({
        "Dataset": f"扩展特征 ({full_feature_dim}维)",
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "AUC": auc,
        "Train Time (s)": train_time
    })
    
    df = pd.DataFrame(results)
    print("\n\n实验2结果对比：")
    print(df.to_string(index=False))
    
    df.to_csv("ablation_experiment_2.csv", index=False)
    
    return df

if __name__ == "__main__":
    # 运行实验1
    df1 = run_experiment_1()
    
    # 运行实验2，假设基础特征维度为42（请根据实际情况调整）
    df2 = run_experiment_2(base_feature_dim=42)