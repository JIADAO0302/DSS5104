import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

def feature_engineering(data):
    #读取数据
    #data = pd.read_csv('house_data.csv', header=0)
    #读取清洗后的数据
    data['date'] = pd.to_datetime(data['date'])
    #数据处理
    data['log_price'] = np.log(data['price'])
    data["yr_after_renovated"] = data['date'].dt.year - data['yr_renovated']
    #遍历所有yr_after_renovated的值，如果yr_after_renovated大于1000,则设置为date的年份-yr_built
    for i in range(len(data)):
        if data['yr_after_renovated'][i] > 1000:
            data['yr_after_renovated'][i] = data['date'].dt.year[i] - data['yr_built'][i]
    data['yr_after_built'] = data['date'].dt.year - data['yr_built']
    data['price_per_sqft'] = data['price'] / data['sqft_living']
    #binary variables
    data['basement_present'] = data['sqft_basement'].apply(lambda x: 1 if x > 0 else 0)
    data['if_renovated'] = data['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)
    #生成if_waterfront的值，如果waterfront大于0,则设置为1，否则设置为0'
    data['if_waterfront'] = data['waterfront'].apply(lambda x: 1 if x > 0 else 0)
    data['basement_ratio'] = data['sqft_basement'] / data['sqft_living']
    data['sqft_above_ratio'] = data['sqft_above'] / data['sqft_living']
    data['rooms'] = data['bedrooms'] + data['bathrooms']

    #对bedrooms、bathrooms、floors进行binning处理，分为1-2, 3-4, 5+三个区间,数值为0,1,2
    data['bedrooms_bin'] = pd.cut(data['bedrooms'], bins=[0, 2, 4, 10], labels=['1-2', '3-4', '5+'])
    data['bathrooms_bin'] = pd.cut(data['bathrooms'], bins=[0, 1, 3, 10], labels=['1', '2-3', '4+'])
    data['floors_bin'] = pd.cut(data['floors'], bins=[0, 1, 2, 10], labels=['1', '2', '3+'])
    #对yr_built进行binning处理，分为1900-1950, 1951-2000, 2001+三个区间,数值为0,1,2
    data['yr_built_bin'] = pd.cut(data['yr_built'], bins=[1900, 1950, 2000, 2023], labels=['1900-1950', '1951-2000', '2001+'])
    #对yr_renovated进行binning处理，分为0-10, 11-20, 21+三个区间,数值为0,1,2
    data['yr_renovated_bin'] = pd.cut(data['yr_renovated'], bins=[0, 10, 20, 100], labels=['0-10', '11-20', '21+'])
    #对sqft_living，sqft_above，sqft_basement进行对数处理
    data['log_sqft_living'] = np.log(data['sqft_living'])
    data['log_sqft_above'] = np.log(data['sqft_above'])
    data['log_sqft_basement'] = np.log(data['sqft_basement'] + 1)
    ### 1. 拆分 statezip 为 state 和 zipcode
    data[['state', 'zipcode']] = data['statezip'].str.extract(r'([A-Z]+)\s+(\d+)', expand=True)
    data['zipcode'] = data['zipcode'].astype(str)

    ### 2. 构造 city / zipcode 的 target encoding 特征
    data['city_avg_price'] = data.groupby('city')['price'].transform('mean')
    data['zipcode_avg_price'] = data.groupby('zipcode')['price'].transform('mean')

    ### 3. 合并稀有城市为 'Other'
    city_counts = data['city'].value_counts()
    rare_cities = city_counts[city_counts < 10].index
    data['city_grouped'] = data['city'].apply(lambda x: 'Other' if x in rare_cities else x)

    ### 4. One-hot 编码城市和州
    data = pd.get_dummies(data, columns=['city_grouped', 'state'], drop_first=True)

    ### 5. 构造 KNN 相似房价特征
    # 选取代表相似性的特征
    knn_features = ['sqft_living', 'bedrooms', 'bathrooms', 'zipcode_avg_price']
    knn_data = data[knn_features].copy()

    # 标准化
    scaler = StandardScaler()
    knn_data_scaled = scaler.fit_transform(knn_data)

    # 建立 KNN 模型（寻找每个房子的 5 个最相似邻居，排除自身）
    knn = NearestNeighbors(n_neighbors=6, algorithm='auto').fit(knn_data_scaled)
    distances, indices = knn.kneighbors(knn_data_scaled)

    # 计算 KNN 平均价格（排除自己）
    knn_avg_prices = []
    for i, neighbor_idxs in enumerate(indices):
        neighbor_prices = data.iloc[neighbor_idxs[1:]]['price']  # 排除自己
        knn_avg_prices.append(neighbor_prices.mean())
    data['knn_avg_price'] = knn_avg_prices

    ### 6. 将 zipcode_avg_price 分为高/中/低三档
    data['zipcode_price_tier'] = pd.qcut(data['zipcode_avg_price'], q=3, labels=['Low', 'Medium', 'High'])

    # One-hot 编码 price tier
    data = pd.get_dummies(data, columns=['zipcode_price_tier'], drop_first=True)
    data.to_csv('engeneered_data.csv', index=False)
    return data