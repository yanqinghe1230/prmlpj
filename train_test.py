import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================================
# 数据集定义
# ============================================================================

class MLPDataset(Dataset):
    """MLP使用：展开所有轨迹为(state, action)对"""
    
    def __init__(self, trajectory_files, train=True,norm_params=None):
        self.states = []
        self.actions = []
        
        for file_path in trajectory_files:
            with open(file_path, 'r') as f:
                trajectory = json.load(f)
            
            for i in range(len(trajectory) - 1):
                # 构建状态向量 (21维)
                state = np.concatenate([
                    trajectory[i]['joint_positions'],      # 7维
                    trajectory[i]['joint_velocities'],     # 7维
                    trajectory[i]['end_effector_position'], # 3维
                    trajectory[i]['end_effector_orientation'] # 4维
                ])
                
                # 动作 = 下一时刻的关节速度
                action = np.array(trajectory[i + 1]['joint_velocities'])
                
                self.states.append(state)
                self.actions.append(action)
        
        self.states = np.array(self.states, dtype=np.float32)
        self.actions = np.array(self.actions, dtype=np.float32)
        
        # 归一化（重要！）
        if norm_params is not None:
            self.state_mean = self.states.mean(axis=0)
            self.state_std = self.states.std(axis=0) + 1e-8
            self.action_mean = self.actions.mean(axis=0)
            self.action_std = self.actions.std(axis=0) + 1e-8
        else:
            self.state_mean = norm_params['state_mean']
            self.state_std = norm_params['state_std']
            self.action_mean = norm_params['action_mean']
            self.action_std = norm_params['action_std']
        
        self.states = (self.states - self.state_mean) / self.state_std
        self.actions = (self.actions - self.action_mean) / self.action_std
        
        print(f"Dataset: {len(self.states)} samples, "
              f"State dim: {self.states.shape[1]}, Action dim: {self.actions.shape[1]}")
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


class LSTMDataset(Dataset):
    """LSTM使用：保持轨迹结构"""
    
    def __init__(self, trajectory_files, history_len=10, train=True):
        """
        Args:
            history_len: 使用多少历史步作为输入
        """
        self.history_len = history_len
        self.trajectories = []
        
        for file_path in trajectory_files:
            with open(file_path, 'r') as f:
                trajectory = json.load(f)
            
            # 提取状态和动作序列
            states = []
            actions = []
            for i in range(len(trajectory) - 1):
                state = np.concatenate([
                    trajectory[i]['joint_positions'],
                    trajectory[i]['joint_velocities'],
                    trajectory[i]['end_effector_position'],
                    trajectory[i]['end_effector_orientation']
                ])
                action = np.array(trajectory[i + 1]['joint_velocities'])
                states.append(state)
                actions.append(action)
            
            self.trajectories.append({
                'states': np.array(states, dtype=np.float32),
                'actions': np.array(actions, dtype=np.float32)
            })
        
        # 归一化
        all_states = np.concatenate([t['states'] for t in self.trajectories])
        all_actions = np.concatenate([t['actions'] for t in self.trajectories])
        
        if train:
            self.state_mean = all_states.mean(axis=0)
            self.state_std = all_states.std(axis=0) + 1e-8
            self.action_mean = all_actions.mean(axis=0)
            self.action_std = all_actions.std(axis=0) + 1e-8
        
        for traj in self.trajectories:
            traj['states'] = (traj['states'] - self.state_mean) / self.state_std
            traj['actions'] = (traj['actions'] - self.action_mean) / self.action_std
        
        # 创建(history, action)对
        self.samples = []
        for traj in self.trajectories:
            for i in range(self.history_len, len(traj['states'])):
                history = traj['states'][i-self.history_len:i]  # (history_len, state_dim)
                action = traj['actions'][i]
                self.samples.append((history, action))
        
        print(f"LSTM Dataset: {len(self.samples)} samples with history_len={history_len}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================================
# 模型定义
# ============================================================================

class MLPPolicy(nn.Module):
    """简单的MLP策略"""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        super().__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, action_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        return self.network(state)


class LSTMPolicy(nn.Module):
    """LSTM策略"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            state_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, state_history):
        """
        Args:
            state_history: (batch, seq_len, state_dim)
        """
        lstm_out, _ = self.lstm(state_history)
        last_hidden = lstm_out[:, -1, :]  # 取最后一个时间步
        action = self.fc(last_hidden)
        return action


# ============================================================================
# 训练函数
# ============================================================================

def train_model(model, train_loader, val_loader, num_epochs=200, lr=1e-3, device='cuda'):
    """通用训练函数"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0
        for batch in train_loader:
            states, actions = batch
            states = states.to(device)
            actions = actions.to(device)
            
            pred_actions = model(states)
            loss = criterion(pred_actions, actions)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                states, actions = batch
                states = states.to(device)
                actions = actions.to(device)
                
                pred_actions = model(states)
                loss = criterion(pred_actions, actions)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'state_mean': train_loader.dataset.state_mean,
                'state_std': train_loader.dataset.state_std,
                'action_mean': train_loader.dataset.action_mean,
                'action_std': train_loader.dataset.action_std,
            }, 'best_model.pth')
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}: '
                  f'Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}')
    
    # 绘制学习曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_curve.png', dpi=150)
    plt.show()
    
    return model, best_val_loss


# ============================================================================
# 主程序
# ============================================================================

def main():
    # 配置
    USE_LSTM = False  # 设为True使用LSTM，False使用MLP
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 加载数据
    trajectory_dir = Path('cleaned_trajectories')  # 你的轨迹文件夹
    trajectory_files = sorted(list(trajectory_dir.glob('*.json')))
    
    print(f"Found {len(trajectory_files)} trajectories")
    
    # 划分训练集和验证集 (80/20)
    split_idx = int(0.8 * len(trajectory_files))
    train_files = trajectory_files[:split_idx]
    val_files = trajectory_files[split_idx:]
    
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
    
    # 创建数据集和加载器
    if USE_LSTM:
        print("\n=== 使用LSTM模型 ===")
        train_dataset = LSTMDataset(train_files, history_len=10, train=True)
        val_dataset = LSTMDataset(val_files, history_len=10, train=False)
        
        # 使用训练集的归一化参数
        val_dataset.state_mean = train_dataset.state_mean
        val_dataset.state_std = train_dataset.state_std
        val_dataset.action_mean = train_dataset.action_mean
        val_dataset.action_std = train_dataset.action_std
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        model = LSTMPolicy(
            state_dim=21,
            action_dim=7,
            hidden_dim=128,
            num_layers=2
        ).to(device)
        
    else:
        print("\n=== 使用MLP模型 ===")
        train_dataset = MLPDataset(train_files, train=True)
        val_dataset = MLPDataset(val_files, train=False)
        
        # 使用训练集的归一化参数
        val_dataset.state_mean = train_dataset.state_mean
        val_dataset.state_std = train_dataset.state_std
        val_dataset.action_mean = train_dataset.action_mean
        val_dataset.action_std = train_dataset.action_std
        
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        
        model = MLPPolicy(
            state_dim=21,
            action_dim=7,
            hidden_dims=[256, 256, 128]
        ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练
    model, best_val_loss = train_model(
        model, 
        train_loader, 
        val_loader,
        num_epochs=200,
        lr=1e-3,
        device=device
    )
    
    print(f"\n训练完成！最佳验证损失: {best_val_loss:.6f}")
    print("模型已保存到 best_model.pth")


if __name__ == '__main__':
    main()