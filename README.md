
# CO₂ 地質封存 PINN v5.2｜封閉儲層模擬儀表板

## 🌐 線上展示

**[點擊開啟 CO₂ PINN v5.2 線上儀表板](https://co2-pinn-v5-2-free7057.streamlit.app)**

老師與同學可以直接使用瀏覽器觀看，不需要安裝 Python 或 Anaconda。

## 📖 專題簡介

本專題使用物理資訊神經網路（Physics-Informed Neural Network, PINN）建立 CO₂ 地質封存儲層的數值預測模型，並透過 Streamlit 製作互動式儀表板，展示不同模擬時間與空間位置的壓力及 CO₂ 飽和度分布。

本版本採用封閉儲層邊界條件的建模設定，目標是探索 PINN 在地下 CO₂ 封存模擬與視覺化上的應用。

## 🔬 主要功能

- 顯示儲層壓力分布（X-Y 剖面）
- 顯示 CO₂ 飽和度分布（X-Z 剖面）
- 互動調整模擬時間、剖面深度及網格精細度
- 載入已訓練的 PINN v5.2 模型
- 提供模型訓練與相關診斷功能
- 支援 OpenUSD 資料匯出，用於三維場景工作流程

## 🛠️ 使用技術

- Python
- PyTorch
- Streamlit
- NumPy
- Plotly
- OpenUSD / NVIDIA Omniverse 工作流程

## 📁 專案檔案

| 檔案 | 說明 |
| --- | --- |
| `dashboard_B_v5_2.py` | Streamlit 儀表板主程式 |
| `co2_s_pinnmodel_v5_2.pth` | PINN v5.2 模型權重 |
| `requirements.txt` | 雲端部署套件清單 |
| `README.md` | 專題介紹 |

## ⚠️ 研究限制

本專題目前屬於研究與教學展示用途。雖然已能載入模型並顯示預測結果，但仍需進一步檢驗初始條件、邊界條件、質量守恆及預測結果的物理合理性。OpenUSD 場景驗證不等同於物理模型驗證。

## 🔗 專題連結

- [線上展示網頁](https://co2-pinn-v5-2-free7057.streamlit.app)
- [GitHub 原始碼](https://github.com/free7057/CO2-PINN-v5-2)
  
