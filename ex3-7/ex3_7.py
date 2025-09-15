import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import glob, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

base_dir = "data"  #یر پوشه اصلی دیتاست
output_file = "3w_merged.parquet"

# بررسی وجود فایل نهایی برای جلوگیری از تکرار
if not os.path.exists(output_file):
    all_files = glob.glob(os.path.join(base_dir, "**/*.csv"), recursive=True)

    # پردازش فایل‌ها یکی‌یکی
    for i, file in enumerate(all_files, 1):
        print(f"Processing {i}/{len(all_files)}: {file}")
        chunk = pd.read_csv(file)

        # ذخیره تدریجی در فایل parquet
        chunk.to_parquet(output_file, engine="pyarrow", index=False)
else:
    print("✅ فایل تجمیع‌شده موجود است:", output_file)

# بارگذاری برای استفاده بعدی
df = pd.read_parquet(output_file)
print("Final shape:", df.shape)
df.head(10)

# -----------------------
# 0) پاک‌سازی داده
#manage NaN value
#df = df.fillna(method='ffill').fillna(method='bfill') 

# تبدیل زمان
df["timestamp"] = pd.to_datetime(df["timestamp"])


# جدا کردن X و y
X = df.drop(columns=["timestamp", "class","P-JUS-CKGL","T-JUS-CKGL","QGL"]).values 
# P-PDG:Permanent Downhole Gauge (PDG)
# P-TPT: Pressure Transducer (TPT),T-TPT: Temperature Transducer (TPT)
# P-MON-CKP: Pressure CKP valve (Choke for Production),T-JUS-CKP:Temprature CKP valve (Choke for Production)

#y = df["class"].values
#print('Unique classes before fix',np.unique(y))

y=df['class'].astype(str).str.replace("^10","",regex=True).astype(int).values
print('Unique classes after fix',np.unique(y))

torch.save((X,y), 'dataset.pt')

X,y = torch.load('dataset.pt')

print(X.shape,y.shape)
print(X)
# -----------------------
# 1) دیتاست
# -----------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_len=50):
        self.X = X
        self.y = y
        self.seq_len = seq_len


    def __len__(self):
        return len(self.X) - self.seq_len


    def __getitem__(self, idx):
        X_seq = self.X[idx:idx+self.seq_len]
        y_label = self.y[idx+self.seq_len]
        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_label, dtype=torch.long)


SEQ_LEN = 50
batch_size = 64
dataset = TimeSeriesDataset(X, y, seq_len=SEQ_LEN)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


n_features = X.shape[1]
#n_classes = len(np.unique(y))
n_classes = int(np.max(y))+1
print('n_classes (by max+1):', n_classes)

# -----------------------
# 2) مدل CNN+LSTM
# -----------------------
class CNN_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, n_classes=2):
        super().__init__()
        # کانولوشن برای استخراج الگوهای محلی
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # LSTM برای وابستگی‌های بلندمدت
        self.lstm = nn.LSTM(32, hidden_dim, num_layers=num_layers, batch_first=True)
        # لایه خروجی
        self.fc = nn.Linear(hidden_dim, n_classes)


    def forward(self, x):
        # x: [batch, seq_len, features]
        x = x.permute(0,2,1)          # [batch, features, seq_len]
        x = self.relu(self.conv1(x))  # [batch, 32, seq_len]
        x = x.permute(0,2,1)          # [batch, seq_len, 32]
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


model = CNN_LSTM(input_dim=n_features, n_classes=n_classes)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# -----------------------
# 3) آموزش با مانیتورینگ
# -----------------------
EPOCHS = 10
losses = []


for epoch in range(EPOCHS):
    total_loss = 0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)


        if torch.isnan(loss):
            print("⚠️ NaN detected in loss. Skipping batch.")
            continue


        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # جلوگیری از انفجار گرادیان
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")


# -----------------------
# 4) نمودار منحنی Loss
# -----------------------
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve (CNN+LSTM)")
plt.show()


# -----------------------
# 5) ارزیابی
# -----------------------
all_preds, all_labels = [], []
model.eval()
with torch.no_grad():
    for X_batch, y_batch in loader:
        preds = model(X_batch)
        all_preds.extend(torch.argmax(preds, dim=1).cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())


print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
print(classification_report(all_labels, all_preds))


cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal","Anomaly"], yticklabels=["Normal","Anomaly"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - CNN+LSTM")
plt.show()

# -----------------------
# 1) انتخاب 5 ستون اصلی
# -----------------------
selected_cols = ["P-PDG","P-TPT","T-TPT","P-MON-CKP","class"]
df_small = df[selected_cols].copy()


# بررسی NaN
print("درصد NaN در هر ستون:\n", df_small.isna().mean())


# پر کردن NaN با میانگین هر ستون
df_small = df_small.fillna(df_small.mean(numeric_only=True))


# -----------------------
# 2) آماده‌سازی X,y
# -----------------------
X = df_small.drop(columns=["class"]).values
y = pd.factorize(df_small["class"])[0]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


print("شکل X:", X_scaled.shape, "شکل y:", y.shape)


# -----------------------
# 3) تحلیل ویژگی‌ها
# -----------------------
# هیستوگرام هر ویژگی بر اساس کلاس
for col in df_small.columns[:-1]:
    plt.figure()
    sns.histplot(data=df_small, x=col, hue="class", bins=30, kde=True, element="step")
    plt.title(f"Histogram of {col} by Class")
    plt.show()


# Boxplot
plt.figure(figsize=(8,5))
df_melt = df_small.melt(id_vars="class", var_name="Feature", value_name="Value")
sns.boxplot(data=df_melt, x="Feature", y="Value", hue="class")
plt.title("Boxplot of Features by Class")
plt.xticks(rotation=45)
plt.show()


# Heatmap همبستگی
plt.figure(figsize=(6,5))
sns.heatmap(df_small.drop(columns="class").corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()


# -----------------------
# 4) مدل پایه Random Forest
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:,1]


print("Random Forest:")
print(classification_report(y_test, y_pred_rf))


cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.show()


# -----------------------
# 5) دیتاست برای PyTorch (سری زمانی)
# -----------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_len=50):
        self.X = X
        self.y = y
        self.seq_len = seq_len
    def __len__(self):
        return len(self.X) - self.seq_len
    def __getitem__(self, idx):
        X_seq = self.X[idx:idx+self.seq_len]
        y_label = self.y[idx+self.seq_len]
        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_label, dtype=torch.long)


SEQ_LEN = 50
dataset = TimeSeriesDataset(X_scaled, y, seq_len=SEQ_LEN)
train_size = int(0.8*len(dataset))
test_size = len(dataset)-train_size
train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)


n_features = X_scaled.shape[1]
n_classes = len(np.unique(y))


# -----------------------
# 6) مدل‌ها
# -----------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, n_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:,-1,:]
        return self.fc(out)


class CNN_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, n_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(32, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_classes)
    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.relu(self.conv1(x))
        x = x.permute(0,2,1)
        out, _ = self.lstm(x)
        out = out[:,-1,:]
        return self.fc(out)


# -----------------------
# 7) حلقه آموزش مشترک
# -----------------------
def train_model(model, train_loader, test_loader, epochs=5, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    all_preds, all_labels, all_probs = [], [], []
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            outputs = model(Xb)
            loss = criterion(outputs, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss/len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss={avg_loss:.4f}")
    # Plot loss
    plt.plot(losses)
    plt.title(f"Loss Curve - {model.__class__.__name__}")
    plt.show()
    # Evaluation
    model.eval()
    with torch.no_grad():
        for Xb,yb in test_loader:
            preds = model(Xb)
            probs = torch.softmax(preds, dim=1)[:,1]
            all_preds.extend(torch.argmax(preds,1).cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    print(classification_report(all_labels, all_preds))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model.__class__.__name__}")
    plt.show()
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc = roc_auc_score(all_labels, all_probs)
    return fpr, tpr, auc


# -----------------------
# 8) اجرا
# -----------------------
print("\n==== LSTM ====")
lstm_model = LSTMModel(n_features, hidden_dim=64, n_classes=n_classes)
fpr_lstm, tpr_lstm, auc_lstm = train_model(lstm_model, train_loader, test_loader)


print("\n==== CNN-LSTM ====")
cnn_lstm_model = CNN_LSTM(n_features, hidden_dim=64, n_classes=n_classes)
fpr_cnn, tpr_cnn, auc_cnn = train_model(cnn_lstm_model, train_loader, test_loader)


# -----------------------
# 9) ROC Curve مقایسه‌ای
# -----------------------
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
auc_rf = roc_auc_score(y_test, y_prob_rf)


plt.figure()
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={auc_rf:.2f})")
plt.plot(fpr_lstm, tpr_lstm, label=f"LSTM (AUC={auc_lstm:.2f})")
plt.plot(fpr_cnn, tpr_cnn, label=f"CNN-LSTM (AUC={auc_cnn:.2f})")
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()



# -----------------------
# 10) PCA دوبعدی
# -----------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


df_pca = pd.DataFrame({
    "PC1": X_pca[:,0],
    "PC2": X_pca[:,1],
    "Class": y
})


plt.figure(figsize=(7,6))
sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="Class", palette="Set1", alpha=0.7)
plt.title("PCA Projection of Data (2D)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Class")
plt.show()
