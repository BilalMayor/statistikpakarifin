import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from scipy import stats
from sklearn.linear_model import LinearRegression

# Load & clean
file_path = r"c:\Users\Bilal Mayor Abdillah\Downloads\Salinan dari data_praktikum_analisis_data - data_praktikum_analisis_data.csv"
df = pd.read_csv(file_path)
df['Total_Sales'] = df['Total_Sales'].fillna(df['Quantity'] * df['Price_Per_Unit'])
df['Order_Date'] = pd.to_datetime(df['Order_Date'])

# ── 1. Underperformer Scatter Plot ──────────────────────────────
prod = df.groupby('Product_Category').agg(Harga=('Price_Per_Unit','mean'), Qty=('Quantity','sum')).reset_index()
prod['Under'] = (prod['Harga'] > prod['Harga'].median()) & (prod['Qty'] < prod['Qty'].median())

plt.figure(figsize=(9,6))
plt.scatter(prod['Qty'], prod['Harga'], c=prod['Under'].map({True:'red',False:'steelblue'}), alpha=0.7, s=80)
for _, row in prod[prod['Under']].iterrows():
    plt.annotate(row['Product_Category'], (row['Qty'], row['Harga']), textcoords='offset points', xytext=(6,4), fontsize=8, color='red')
plt.axvline(prod['Qty'].median(), color='gray', linestyle='--')
plt.axhline(prod['Harga'].median(), color='gray', linestyle=':')
plt.title('Produk Underperformer (Merah = Mahal & Jarang Laku)')
plt.xlabel('Total Kuantitas'); plt.ylabel('Rata-rata Harga')
plt.tight_layout(); plt.savefig('1_underperformer.png'); plt.close()
print("1. Selesai → 1_underperformer.png")

# ── 2. Segmentasi Pelanggan RFM Dasar ──────────────────────────
snap = df['Order_Date'].max() + dt.timedelta(days=1)
rfm = df.groupby('CustomerID').agg(
    Recency=('Order_Date', lambda x: (snap - x.max()).days),
    Frequency=('Order_ID', 'count'),
    Monetary=('Total_Sales', 'sum')
).reset_index()
rfm['R'] = pd.qcut(rfm['Recency'].rank(method='first'),   5, labels=[5,4,3,2,1]).astype(int)
rfm['F'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
rfm['M'] = pd.qcut(rfm['Monetary'].rank(method='first'),  5, labels=[1,2,3,4,5]).astype(int)
rfm['Score'] = rfm['R'] + rfm['F'] + rfm['M']
rfm['Segment'] = pd.cut(rfm['Score'], bins=[2,6,9,12,15], labels=['Lost','At Risk','Loyal','Champions'])

seg_count = rfm['Segment'].value_counts()
seg_count.plot(kind='barh', color=['#e74c3c','#e67e22','#27ae60','#2ecc71'][:len(seg_count)], figsize=(8,5))
plt.title('Segmentasi Pelanggan RFM'); plt.xlabel('Jumlah Pelanggan')
plt.tight_layout(); plt.savefig('2_segmentasi_rfm.png'); plt.close()
print("2. Selesai → 2_segmentasi_rfm.png")

# ── 3. Kontribusi Kategori (Bar Horizontal) ────────────────────
cat = df.groupby('Product_Category').agg(Revenue=('Total_Sales','sum'), AdBudget=('Ad_Budget','sum')).sort_values('Revenue')
fig, ax = plt.subplots(figsize=(9,6))
y = np.arange(len(cat))
ax.barh(y+0.2, cat['Revenue'],  0.4, label='Revenue',   color='steelblue')
ax.barh(y-0.2, cat['AdBudget'], 0.4, label='Ad Budget', color='orange')
ax.set_yticks(y); ax.set_yticklabels(cat.index)
ax.set_title('Revenue vs Ad Budget per Kategori'); ax.legend()
plt.tight_layout(); plt.savefig('3_kontribusi_kategori.png'); plt.close()
print("3. Selesai → 3_kontribusi_kategori.png")

# ── 4. Uji Hipotesis Iklan vs Penjualan ───────────────────────
med = df['Ad_Budget'].median()
t, p = stats.ttest_ind(df[df['Ad_Budget'] >= med]['Total_Sales'],
                        df[df['Ad_Budget'] <  med]['Total_Sales'], equal_var=False)
print(f"4. T-test: t={t:.3f}, p={p:.4f} → {'Signifikan' if p<0.05 else 'Tidak Signifikan'}")

# ── 5. RFM Detail Bubble Chart ────────────────────────────────
colors = rfm['Segment'].map({'Champions':'#2ecc71','Loyal':'#27ae60','At Risk':'#e67e22','Lost':'#e74c3c'})
plt.figure(figsize=(9,6))
plt.scatter(rfm['Frequency'], rfm['Monetary'],
            s=np.clip(500/(rfm['Recency']+1)*30, 20, 400),
            c=colors, alpha=0.6, edgecolors='white')
plt.title('RFM Detail: Frequency vs Monetary (ukuran = Recency)')
plt.xlabel('Frequency'); plt.ylabel('Monetary')
plt.tight_layout(); plt.savefig('5_rfm_detail.png'); plt.close()
print("5. Selesai → 5_rfm_detail.png")

# ── 6. Regresi Linear Ad_Budget → Total_Sales ─────────────────
model = LinearRegression().fit(df[['Ad_Budget']], df['Total_Sales'])
r2 = model.score(df[['Ad_Budget']], df['Total_Sales'])
print(f"6. Regresi: slope={model.coef_[0]:.4f}, R²={r2:.4f}")

x_line = np.linspace(df['Ad_Budget'].min(), df['Ad_Budget'].max(), 100).reshape(-1,1)
plt.figure(figsize=(8,5))
plt.scatter(df['Ad_Budget'], df['Total_Sales'], alpha=0.4, s=20, color='steelblue')
plt.plot(x_line, model.predict(x_line), color='red', linewidth=2, label=f'R²={r2:.3f}')
plt.title('Regresi Linear: Ad_Budget → Total_Sales')
plt.xlabel('Ad Budget'); plt.ylabel('Total Sales'); plt.legend()
plt.tight_layout(); plt.savefig('6_regresi.png'); plt.close()
print("6. Selesai → 6_regresi.png")

print("\n✅ Semua analisis selesai!")