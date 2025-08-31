# نصب کتابخانه‌ها (فقط بار اول لازم است)
# !pip install torch torchvision arabic-reshaper python-bidi seaborn openpyxl


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


import arabic_reshaper
from bidi.algorithm import get_display


# =======================
# 1. خواندن دیتاست
# =======================
df = pd.read_excel("DistillationColumn.xlsx")
print("ابعاد داده:", df.shape)
print(df.head())


# =======================
# 2. بررسی اولیه و پاکسازی
# =======================
print(df.info())
print(df.describe().T)


# پر کردن مقادیر گمشده
df = df.fillna(method="ffill").fillna(method="bfill")


# =======================
# 3. تحلیل اکتشافی
# =======================
def fa_text(text):
    return get_display(arabic_reshaper.reshape(text))


# توزیع داده‌ها
df.hist(figsize=(15, 10), bins=30)
plt.suptitle(fa_text("توزیع داده‌های حسگرها"))
plt.show()


# همبستگی
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title(fa_text("نقشه همبستگی بین متغیرها"))
plt.show()


# =======================
# 4. پیش پردازش
# =======================
target_col = df.columns[-1]   # فرض: آخرین ستون خروجی است
sensor_cols = df.columns[:-1]


X_raw = df[sensor_cols].values
y_raw = df[target_col].values


# نرمال‌سازی
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)


# پنجره‌های زمانی
window_size = 10
def make_windows(X, y, window):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i+window].flatten())
        ys.append(y[i+window])
    return np.array(Xs), np.array(ys)


X, y = make_windows(X_scaled, y_raw, window_size)


# تقسیم داده
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)


# تبدیل به Tensor
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
X_val_t   = torch.tensor(X_val, dtype=torch.float32)
y_val_t   = torch.tensor(y_val, dtype=torch.float32).view(-1,1)


train_ds = TensorDataset(X_train_t, y_train_t)
val_ds   = TensorDataset(X_val_t, y_val_t)


train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)


# =======================
# 5. مدل MLP در PyTorch
# =======================
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


input_dim = X_train.shape[1]
model = MLP(input_dim)


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# =======================
# 6. آموزش
# =======================
n_epochs = 100
train_losses, val_losses = [], []


for epoch in range(n_epochs):
    model.train()
    batch_losses = []
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    train_loss = np.mean(batch_losses)


    # اعتبارسنجی
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t)
        val_loss = criterion(val_preds, y_val_t).item()
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)


    if epoch % 10 == 0:
        print(f"دور {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")


# =======================
# 7. ارزیابی
# =======================
model.eval()
with torch.no_grad():
    y_pred = model(X_val_t).squeeze().numpy()


rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mae = mean_absolute_error(y_val, y_pred)
mape = mean_absolute_percentage_error(y_val, y_pred)


print("RMSE:", rmse)
print("MAE:", mae)
print("MAPE:", mape)


# =======================
# 8. نمودارها
# =======================
# روند آموزش
plt.figure(figsize=(10,5))
plt.plot(train_losses, label=fa_text("خطای آموزش"))
plt.plot(val_losses, label=fa_text("خطای اعتبارسنجی"))
plt.legend()
plt.title(fa_text("روند آموزش مدل MLP (PyTorch)"))
plt.xlabel(fa_text("دور (epoch)"))
plt.ylabel(fa_text("مقدار خطا"))
plt.show()


# واقعی vs پیش‌بینی
plt.figure(figsize=(12,6))
plt.plot(y_val[:200], label=fa_text("واقعی"))
plt.plot(y_pred[:200], label=fa_text("پیش‌بینی"))
plt.legend()
plt.title(fa_text("مقایسه مقدار واقعی و پیش‌بینی شده"))
plt.xlabel(fa_text("شماره نمونه"))
plt.ylabel(fa_text("خروجی"))
plt.show()


# نمودار پراکندگی
plt.figure(figsize=(7,7))
sns.scatterplot(x=y_val, y=y_pred, alpha=0.6)
plt.xlabel(fa_text("واقعی"))
plt.ylabel(fa_text("پیش‌بینی"))
plt.title(fa_text("پراکندگی واقعی در مقابل پیش‌بینی"))
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.show()
