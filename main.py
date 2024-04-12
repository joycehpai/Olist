import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

dfs_list = {
    'customer_df':pd.read_csv("/Users/joycepai/PycharmProjects/1/pythonProject/OlisCustomerProject/olist_customers_dataset.csv"),
    'order_items_df': pd.read_csv("/Users/joycepai/PycharmProjects/1/pythonProject/OlisCustomerProject/olist_order_items_dataset.csv"),
    'payment_df': pd.read_csv("/Users/joycepai/PycharmProjects/1/pythonProject/OlisCustomerProject/olist_order_payments_dataset.csv"),
    'reviews_df': pd.read_csv("/Users/joycepai/PycharmProjects/1/pythonProject/OlisCustomerProject/olist_order_reviews_dataset.csv"),
    'orders_df': pd.read_csv("/Users/joycepai/PycharmProjects/1/pythonProject/OlisCustomerProject/olist_orders_dataset.csv"),
    'products_df': pd.read_csv("/Users/joycepai/PycharmProjects/1/pythonProject/OlisCustomerProject/olist_products_dataset.csv")
}
# 清洗資料
# def check():
    # for df in dfs_list:
pd.set_option('Display.max_columns',None)
# print(dfs_list['customer_df'].head())
# print(dfs_list['customer_df'].info())
# print(dfs_list['payment_df'].head())
# print(dfs_list['payment_df'].info())
# print(dfs_list['orders_df'].head())
# print(dfs_list['orders_df'].info())
# print(dfs_list['order_items_df'].head())
# print(dfs_list['order_items_df'].info())
# print(dfs_list['products_df'].head())
# print(dfs_list['products_df'].info())

# customer_df
# 沒有缺失值 99441筆
# 郵遞區號、城市、州

# order_items_df
# 112650筆資料，沒有缺失值
# 調整object為date
dfs_list['order_items_df']['shipping_limit_date'] = pd.to_datetime(dfs_list['order_items_df']['shipping_limit_date'],
                                                                   format='%Y-%m-%d %H:%M:%S')
# 只截取order_id,order_item_id,product_id,shipping_limit_date,price,freight_value
order_items_v2 = dfs_list['order_items_df'][['order_id','order_item_id','product_id','shipping_limit_date','price'
    ,'freight_value']]
# payment_df
# payment_sequential表示的是同一個訂單中的支付序列
# payment_installments表示的是客戶進行分期付款的次數
# 103886筆，沒有缺失值

# reviews_df

# orders_df
# 總共99441筆
# order_delivered_carrier_date表示訂單已被運送公司接收的日期。
# order_delivered_customer_date表示訂單被實際交付給客戶的日期
# order_estimated_delivery_date表示 Olist 預估的訂單交付日期
# 調整object轉為date
dfs_list['orders_df']['order_delivered_carrier_date'] = pd.to_datetime(dfs_list['orders_df']['order_delivered_carrier_date'],
                                                                   format='%Y-%m-%d %H:%M:%S')
dfs_list['orders_df']['order_delivered_customer_date'] = pd.to_datetime(dfs_list['orders_df']['order_delivered_customer_date'],
                                                                   format='%Y-%m-%d %H:%M:%S')
dfs_list['orders_df']['order_estimated_delivery_date'] = pd.to_datetime(dfs_list['orders_df']['order_estimated_delivery_date'],
                                                                   format='%Y-%m-%d %H:%M:%S')
# 比較時點
# order_delivered_carrier_date與order_delivered_customer_date的比較，知道貨運公司通常接收到交付需要多少時間
orders_df = dfs_list['orders_df']
orders_df['days_needed_for_delivery'] = (orders_df['order_delivered_customer_date'] - orders_df['order_delivered_carrier_date']).dt.days
# print(orders_df['days_needed_for_delivery'])
time_for_delivery = orders_df.groupby('days_needed_for_delivery')['days_needed_for_delivery'].count()
# print(time_for_delivery)
# order_delivered_customer_date和order_estimated_delivery_date算一個KPI，分為提前、如期、跟延遲，並計算比例
orders_df['delivery_difference'] = orders_df['order_delivered_customer_date'] - orders_df['order_estimated_delivery_date']
def categorize_delivery(diff):
    if diff.days < 0:
        return 'earlier'
    elif diff.days == 0:
        return 'on time'
    else:
        return 'late'

orders_df['delivery_category'] = orders_df['delivery_difference'].apply(categorize_delivery)
delivery_counts = orders_df['delivery_category'].value_counts()
delivery_percentages = orders_df['delivery_category'].value_counts(normalize=True) * 100
result_df = pd.DataFrame({'Count': delivery_counts, 'Percentage (%)': delivery_percentages})
# print(result_df)

# products_df
# 32951筆資料，僅32341筆類別資料，沒有商品名稱，所以取product_id和product_category_name即可
products_df_v2 = dfs_list['products_df'][['product_id','product_category_name']]
# check()

# 合併需要的表單
df1 = pd.merge(dfs_list['orders_df'],dfs_list['payment_df'],on='order_id',how="left")
df2 = pd.merge(df1,dfs_list['customer_df'],on='customer_id',how="left")
df3 = pd.merge(df2,order_items_v2,on="order_id",how="left")
df = pd.merge(df3,products_df_v2,on="product_id",how="left")
# print(df.head())
# print(df.info())
# print(df.shape)
# print(df.columns)
# 付款行為分析
# customer_unique_id,order_purchase_timestamp,payment_type,payment_installments,payment_value
payment_analysis_df = df[['customer_unique_id','order_id','order_purchase_timestamp','payment_type',
                          'payment_installments','payment_value']]
# 分析顧客的支付方式偏好。您可以計算每種支付方式的使用頻率和總支付金額。
payment_type_counts = (payment_analysis_df.groupby('payment_type').agg({'customer_unique_id':'count',
                                                                        'payment_value':'sum'
                                                                        }).reset_index()
                       .sort_values(by='customer_unique_id',ascending=False))
payment_type_counts['payment_values_percentage'] = round(payment_type_counts['payment_value'] / payment_type_counts['payment_value'].sum()*100,2)
# print(payment_type_counts)
# 先把credit card的人找出來
credit_card_focus = payment_analysis_df[payment_analysis_df['payment_type'] == 'credit_card']
# 分期付款的期數及金額的關係
payment_installments_counts = credit_card_focus.groupby('payment_installments').agg({'payment_installments':'count',
                                                                                     'payment_value':['sum','mean']})
# print(payment_installments_counts)
# 用數字證實金額越高，分期付款次數越多
# 顧客下訂次數
order_counts = payment_analysis_df.groupby('customer_unique_id')['order_id'].nunique()
# print(order_counts)
# 所有可識別顧客僅下訂一次

# 顧客群體分析：
# 根據RFM將顧客分群。這可以幫助您更好地理解不同顧客群體的特徵和需求。
# 計算 Recency
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
last_purchase_date = df['order_purchase_timestamp'].max()
df['Recency'] = (last_purchase_date - df['order_purchase_timestamp']).dt.days

# 計算 Frequency
frequency_df = df.groupby('customer_unique_id')['order_id'].count().reset_index()
frequency_df.columns = ['customer_unique_id', 'Frequency']

# 計算 Monetary
monetary_df = df.groupby('customer_unique_id')['payment_value'].sum().reset_index()
monetary_df.columns = ['customer_unique_id', 'Monetary']

# 合併 Recency、Frequency 和 Monetary
rfm_df = pd.merge(pd.merge(frequency_df, monetary_df, on='customer_unique_id'),
                  df[['customer_unique_id', 'Recency']],
                  on='customer_unique_id')
# print(rfm_df)
# 將客戶分為幾個 RFM 群組
quantiles = rfm_df[['Recency', 'Frequency', 'Monetary']].quantile(q=[0.25, 0.5, 0.75])
quantiles = quantiles.to_dict()

def r_score(x, p, d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]:
        return 2
    else:
        return 1

def fm_score(x, p, d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]:
        return 3
    else:
        return 4

rfm_df['R'] = rfm_df['Recency'].apply(r_score, args=('Recency', quantiles))
rfm_df['F'] = rfm_df['Frequency'].apply(fm_score, args=('Frequency', quantiles))
rfm_df['M'] = rfm_df['Monetary'].apply(fm_score, args=('Monetary', quantiles))

# 組合 RFM 得分
rfm_df['RFM_Score'] = rfm_df['R'].astype(str) + rfm_df['F'].astype(str) + rfm_df['M'].astype(str)

# 分析 RFM 群組
rfm_grouped = rfm_df.groupby('RFM_Score').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'count']
}).round(1)

# 列重命名
rfm_grouped.columns = ['RecencyMean', 'FrequencyMean', 'MonetaryMean', 'Count']

# print(rfm_grouped)

# 機器學習的顧客分群
# 選擇 RFM 作為特徵
X = rfm_df[['Recency', 'Frequency', 'Monetary']]

# 標準化特徵
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 計算肘部值（Elbow Method）來選擇 K
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o', linestyle='--')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# 根據肘部法則選擇 K 值
# 假設選擇 K=5
k = 5

# 擬合 K-means 模型
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)

# 將分群結果添加到 DataFrame
rfm_df['Cluster'] = kmeans.labels_
# print(rfm_df)
# 分析每個集群的特徵
cluster_means = rfm_df.groupby('Cluster')[['R','F','M']].mean()

print(cluster_means)
# 考慮使用機器學習模型來預測未來營收。


