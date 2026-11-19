import os
import json
import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score

os.makedirs('prediksi/ml_models', exist_ok=True)
os.makedirs('prediksi/static/prediksi', exist_ok=True)

url = "https://raw.githubusercontent.com/asaputro281-boo/All-Project/main/laptop_price.csv"
df = pd.read_csv(url, encoding='latin-1')

df['Ram_GB'] = df['Ram'].astype(str).str.replace('GB', '', regex=False).astype(int)

def parse_storage(mem_str):
    total_gb = 0
    for part in str(mem_str).upper().split('+'):
        part = part.strip()
        num = ''.join([c for c in part.split()[0] if c.isdigit() or c == '.'])
        if num:
            val = float(num) * (1024 if 'TB' in part else 1)
            total_gb += val
    return int(total_gb) if total_gb > 0 else 256

def parse_cpu_tier(cpu_str):
    cpu_str = str(cpu_str).lower()
    if 'i9' in cpu_str or 'ryzen 9' in cpu_str or 'xeon' in cpu_str: return 5
    elif 'i7' in cpu_str or 'ryzen 7' in cpu_str: return 4
    elif 'i5' in cpu_str or 'ryzen 5' in cpu_str: return 3
    elif 'i3' in cpu_str or 'ryzen 3' in cpu_str: return 2
    return 1

df['Storage_GB'] = df['Memory'].apply(parse_storage)
df['CPU_Tier'] = df['Cpu'].apply(parse_cpu_tier)
target_col = 'Price_euros' if 'Price_euros' in df.columns else df.columns[-1]

X = df[['Ram_GB', 'Storage_GB', 'CPU_Tier']]
y_reg = df[target_col]
y_clf = pd.qcut(df[target_col], q=3, labels=['Murah', 'Sedang', 'Mahal'])

X_train, X_test, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
_, _, y_train_c, y_test_c = train_test_split(X, y_clf, test_size=0.2, random_state=42)

model_reg = LinearRegression().fit(X_train, y_train_r)
model_dt = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X_train, y_train_c)

metrics = {
    'r2_score': round(r2_score(y_test_r, model_reg.predict(X_test)), 3),
    'mae': round(mean_absolute_error(y_test_r, model_reg.predict(X_test)), 2),
    'accuracy': round(accuracy_score(y_test_c, model_dt.predict(X_test)) * 100, 2),
    'total_data': len(df)
}

model_reg.fit(X, y_reg)
model_dt.fit(X, y_clf)

joblib.dump(model_reg, 'prediksi/ml_models/model_regresi.pkl')
joblib.dump(model_dt, 'prediksi/ml_models/model_dt.pkl')
with open('prediksi/ml_models/metrics.json', 'w') as f:
    json.dump(metrics, f)

plt.figure(figsize=(8, 4))
plt.scatter(df['Ram_GB'], y_reg, color='#2563eb', alpha=0.35, label='Harga Aktual Dataset')
plt.scatter(df['Ram_GB'], model_reg.predict(X), color='#dc2626', alpha=0.6, label='Garis Prediksi Regresi')
plt.title('Visualisasi Regresi: Kapasitas RAM vs Harga Laptop')
plt.xlabel('RAM (GB)')
plt.ylabel('Harga (EUR)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('prediksi/static/prediksi/grafik_regresi.png')

print("Sukses! Model ML, metrik evaluasi, dan grafik berhasil dibuat.")