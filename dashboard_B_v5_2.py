###########################################################################
# [PINN v5.2 TEST] Closed-boundary CO2 Storage PINN + Streamlit Dashboard
#
# 特點:
# - 壓力改為「無量綱」輸出 p_hat，物理解算時再乘 P_REF
# - 兩階段訓練：
#     Stage 1: 只學初始條件 (IC)
#     Stage 2: IC + PDE 殘差 (PDE Loss 權重強化)
# - 自動判斷裝置：支援正常 CUDA GPU；對 SM>=10 (如 RTX 5090) 強制使用 CPU
#
# Author: Simon Su PhD., Skybits
# v5 experimental adaptation based on original v4
###########################################################################

import torch
import torch.nn as nn
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from math import erf, sqrt, pi
from datetime import datetime
from pathlib import Path

# -------------------------------------------------------------------------
# 0. Streamlit 基本設定
# -------------------------------------------------------------------------
st.set_page_config(layout="wide")
st.title("CO2 保存物理資訊 PINN v5.2 封閉儲層儀錶板")

# -------------------------------------------------------------------------
# 1. 裝置選擇：保留 CUDA 給舊架構 GPU，但對 RTX 5090 這類新架構強制用 CPU
# -------------------------------------------------------------------------
def select_device():
    """
    安全選擇運算裝置：
    - 若有 CUDA，先看 compute capability
      * SM >= 10 (例如 RTX 5090, Blackwell) → PyTorch Windows 尚未正式支援 → 強制 CPU
      * SM < 10 (30/40 系列等) → 嘗試使用 CUDA
    - 若 CUDA 初始化失敗 → 改用 CPU
    """
    if torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability()
        except Exception:
            major, minor = (0, 0)

        # Blackwell / 未支援新架構（如 SM 120）→ 強制用 CPU
        if major >= 10:
            st.sidebar.warning(
                f"偵測到 GPU Compute Capability = SM{major}{minor} "
                f"（例如 RTX 5090 這類新架構），目前 PyTorch Windows 版本"
                f"對此支援不完整，為避免不穩定，已強制使用 CPU。"
            )
            return "cpu"

        # 舊架構 (<= Hopper) 嘗試使用 CUDA
        try:
            _ = torch.randn(1).to("cuda")
            st.sidebar.success(
                f"使用 CUDA GPU (Compute Capability SM{major}{minor}) 進行訓練。"
            )
            return "cuda"
        except Exception as e:
            st.sidebar.warning(
                f"偵測到 CUDA 但初始化失敗，改用 CPU。\n詳細訊息: {e}"
            )
            return "cpu"

    st.sidebar.info("未偵測到 CUDA，使用 CPU 執行。")
    return "cpu"


DEVICE = select_device()
print(f"[PINN v5.2] Using device: {DEVICE}")

# -------------------------------------------------------------------------
# 2. 物理參數與領域定義 (SI 單位)
# -------------------------------------------------------------------------
GRAVITY = 9.81              # m/s^2
WATER_DENSITY = 1050.0      # kg/m^3
CO2_DENSITY_SC = 700.0      # kg/m^3
WATER_VISCOSITY = 8.9e-4    # Pa·s
CO2_VISCOSITY_SC = 7.5e-5   # Pa·s
POROSITY = 0.25
PERMEABILITY = 5e-14        # m^2
TOTAL_COMPRESSIBILITY = 1e-9  # Pa^-1; 示範有效儲存係數，非實測
SOURCE_SIGMA = 50.0          # m
PRESSURE_SCALE = 1e7         # Pa, 殘差尺度
TIME_SCALE = T_SCALE = 3650.0 * 86400.0  # s
FLOW_SCALE = 1e-7            # m/s, BC 殘差尺度
RESIDUAL_SCALE = 1e-8        # s^-1, PDE 殘差尺度

# 空間範圍 (m)
X_MIN, X_MAX = -1500.0, 1500.0
Y_MIN, Y_MAX = -1500.0, 1500.0
Z_INJECTION_DEPTH = -1800.0
Z_MIN, Z_MAX = Z_INJECTION_DEPTH - 200.0, Z_INJECTION_DEPTH + 200.0

# 時間範圍 (s)：0 ~ 10 年
T_MIN, T_MAX = 0.0, 3650.0 * 86400.0

# 每口井: (x, y, z, rate_kg_s)
WELLS = [
    (-300.0, 0.0, Z_INJECTION_DEPTH, 15.0),  # 井 1
    ( 300.0, 0.0, Z_INJECTION_DEPTH, 15.0),  # 井 2
]


# Domain 正規化用 mean/scale (numpy)
domain_means = np.array(
    [
        (X_MIN + X_MAX) / 2.0,
        (Y_MIN + Y_MAX) / 2.0,
        (Z_MIN + Z_MAX) / 2.0,
        (T_MIN + T_MAX) / 2.0,
    ]
)
domain_scales = np.array(
    [
        (X_MAX - X_MIN) / 2.0,
        (Y_MAX - Y_MIN) / 2.0,
        (Z_MAX - Z_MIN) / 2.0,
        (T_MAX - T_MIN) / 2.0,
    ]
)

# 壓力參考值 (Pa)：用來把壓力無量綱化
P_REF = 1e7  # 10 MPa 級別

MODEL_PATH = Path(__file__).resolve().parent / "co2_s_pinnmodel_v5_2.pth"  # 獨立 v5 權重

# -------------------------------------------------------------------------
# 3. Streamlit 側邊欄控制
# -------------------------------------------------------------------------
st.sidebar.header("模型與訓練控制")

time_slider = st.sidebar.slider("選擇可視化時間 (天)", 0.0, 3650.0, 100.0, 10.0)
z_slice = st.sidebar.slider(
    "選擇切面深度 z (m)",
    float(Z_MIN),
    float(Z_MAX),
    float(Z_INJECTION_DEPTH),
    10.0
)
N_POINTS_VIS = st.sidebar.select_slider(
    "可視化網格精細度", options=[16, 32, 64, 100], value=64
)

# 訓練設定依裝置給不同預設
if DEVICE == "cuda":
    default_epochs_ic = 200
    default_epochs_full = 800
    default_N_pde = 4096
    default_N_ic = 1024
else:
    default_epochs_ic = 100
    default_epochs_full = 400
    default_N_pde = 2048
    default_N_ic = 512

epochs_ic = st.sidebar.number_input(
    "Stage 1 (只學初始條件 IC) 的 Epoch 數",
    min_value=20,
    max_value=2000,
    value=default_epochs_ic,
    step=20,
)
epochs_full = st.sidebar.number_input(
    "Stage 2 (IC + PDE) 的 Epoch 數",
    min_value=50,
    max_value=5000,
    value=default_epochs_full,
    step=50,
)
N_PDE_SAMPLES = st.sidebar.number_input(
    "每次 PDE 采樣點數 N_pde",
    min_value=512,
    max_value=8192,
    value=default_N_pde,
    step=512,
)
N_IC_SAMPLES = st.sidebar.number_input(
    "每次初始條件采樣點數 N_ic",
    min_value=256,
    max_value=4096,
    value=default_N_ic,
    step=256,
)

# -------------------------------------------------------------------------
# 4. Domain Tensor (避免每次重建)
# -------------------------------------------------------------------------
@st.cache_resource
def get_domain_tensors(device: str):
    dm = torch.tensor(domain_means, dtype=torch.float32, device=device)
    ds = torch.tensor(domain_scales, dtype=torch.float32, device=device)
    return dm, ds


DOMAIN_MEANS_TENSOR, DOMAIN_SCALES_TENSOR = get_domain_tensors(DEVICE)

# -------------------------------------------------------------------------
# 5. PINN Model 定義 (v5: 壓力輸出為無量綱 p_hat)
# -------------------------------------------------------------------------
class PINNSolver(nn.Module):
    """
    輸出:
      p_hat: 無量綱壓力，實際壓力 p = p_hat * P_REF (Pa)
      S_g  : CO2 氣相飽和度 ∈ [0, 1]
    """
    def __init__(self, in_dim=4, hidden_dim=96, num_hidden_layers=4, out_dim=2):
        super(PINNSolver, self).__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.network = nn.Sequential(*layers)
        # 未訓練模型的 CO2 飽和度先接近零，而非 sigmoid(0)=0.5。
        # 僅影響全新 v5.2 權重的初始化，不預設井附近已有 CO2 羽流。
        with torch.no_grad():
            self.network[-1].bias[0] = 1.85
            self.network[-1].bias[1] = -4.0

    def forward(self, x_norm, y_norm, z_norm, t_norm):
        inputs = torch.cat([x_norm, y_norm, z_norm, t_norm], dim=1)  # (N, 4)
        outputs = self.network(inputs)  # (N, 2)
        p_hat = outputs[:, 0:1]       # 無量綱壓力
        Sg_raw = outputs[:, 1:2]      # CO2 飽和度 logits
        S_g = torch.sigmoid(Sg_raw)   # 限制在 [0, 1]
        return p_hat, S_g

# -------------------------------------------------------------------------
# 6. PDE 殘差計算 (在物理維度上計算壓力 p = p_hat * P_REF)
# -------------------------------------------------------------------------
def model_physical(model, x, y, z, t):
    return model((x-DOMAIN_MEANS_TENSOR[0])/DOMAIN_SCALES_TENSOR[0],
                 (y-DOMAIN_MEANS_TENSOR[1])/DOMAIN_SCALES_TENSOR[1],
                 (z-DOMAIN_MEANS_TENSOR[2])/DOMAIN_SCALES_TENSOR[2],
                 (t-DOMAIN_MEANS_TENSOR[3])/DOMAIN_SCALES_TENSOR[3])


def gaussian_interval_fraction(lo, hi, mu, sigma):
    return 0.5 * (erf((hi-mu)/(sqrt(2)*sigma)) - erf((lo-mu)/(sqrt(2)*sigma)))


def source_normalization(wx, wy, wz, sigma):
    # 有限儲層六面截斷後仍維持每口井的指定注入量 (m3/s)
    fraction = (gaussian_interval_fraction(X_MIN,X_MAX,wx,sigma)
                * gaussian_interval_fraction(Y_MIN,Y_MAX,wy,sigma)
                * gaussian_interval_fraction(Z_MIN,Z_MAX,wz,sigma))
    return (2*pi)**1.5 * sigma**3 * fraction


def injection_source(x, y, z):
    # volumetric source q [1/s], volume integral equals sum(rate / density) [m3/s]
    q = torch.zeros_like(x)
    sigma = SOURCE_SIGMA
    for wx, wy, wz, rate in WELLS:
        norm = source_normalization(wx,wy,wz,sigma)
        r2 = (x-wx)**2 + (y-wy)**2 + (z-wz)**2
        q = q + (rate / CO2_DENSITY_SC) * torch.exp(-r2/(2*sigma*sigma)) / norm
    return q


def phase_fluxes(model, x, y, z, t):
    for a in (x,y,z,t):
        a.requires_grad_(True)
    p_hat, sg = model_physical(model,x,y,z,t)
    p = p_hat*P_REF
    px,py,pz = torch.autograd.grad(p.sum(),(x,y,z),create_graph=True)
    # z 為向上的座標，靜水壓 p_z=-rho*g；故重力項需 +rho*g
    # Corey 型相對滲透率（無殘餘飽和度的示範假設）
    krg = sg**2
    krw = (1-sg)**2
    uw = tuple(-PERMEABILITY*krw/WATER_VISCOSITY * a for a in (px,py,pz+WATER_DENSITY*GRAVITY))
    ug = tuple(-PERMEABILITY*krg/CO2_VISCOSITY_SC * a for a in (px,py,pz+CO2_DENSITY_SC*GRAVITY))
    return p,sg,uw,ug


def compute_residuals(model,x,y,z,t):
    p,sg,uw,ug = phase_fluxes(model,x,y,z,t)
    pt = torch.autograd.grad(p.sum(),t,create_graph=True)[0]
    sgt = torch.autograd.grad(sg.sum(),t,create_graph=True)[0]
    div_total = sum(torch.autograd.grad((uw[i]+ug[i]).sum(),coord,create_graph=True)[0]
                    for i,coord in enumerate((x,y,z)))
    div_gas = sum(torch.autograd.grad(ug[i].sum(),coord,create_graph=True)[0]
                  for i,coord in enumerate((x,y,z)))
    q = injection_source(x,y,z)
    # 示範弱可壓縮總體積方程 + 定密度氣相飽和度方程
    eq_pressure = TOTAL_COMPRESSIBILITY*pt + div_total - q
    eq_gas = POROSITY*sgt + div_gas - q
    return eq_pressure/RESIDUAL_SCALE, eq_gas/RESIDUAL_SCALE


def sample_pde(n,device):
    # 1/4 全域 + 1/4 井周 sigma + 1/4 井周 2sigma + 1/4 井周 4sigma
    # 用於局部 PDE 殘差，不可直接用此非均勻樣本估算全域積分。
    def uniform(k,lo,hi): return torch.rand(k,1,device=device)*(hi-lo)+lo
    counts=[n//4,n//4,n//4,n-3*(n//4)]
    xs=[uniform(counts[0],X_MIN,X_MAX)]
    ys=[uniform(counts[0],Y_MIN,Y_MAX)]
    zs=[uniform(counts[0],Z_MIN,Z_MAX)]
    for k,scale in zip(counts[1:],(1.0,2.0,4.0)):
        ids=torch.randint(len(WELLS),(k,),device=device)
        wx=torch.tensor([w[0] for w in WELLS],device=device)[ids].view(-1,1)
        wy=torch.tensor([w[1] for w in WELLS],device=device)[ids].view(-1,1)
        wz=torch.tensor([w[2] for w in WELLS],device=device)[ids].view(-1,1)
        xs.append((wx+torch.randn(k,1,device=device)*SOURCE_SIGMA*scale).clamp(X_MIN,X_MAX))
        ys.append((wy+torch.randn(k,1,device=device)*SOURCE_SIGMA*scale).clamp(Y_MIN,Y_MAX))
        zs.append((wz+torch.randn(k,1,device=device)*SOURCE_SIGMA*scale).clamp(Z_MIN,Z_MAX))
    n_early=n//2
    t=torch.cat((uniform(n_early,T_MIN,365*86400.0),uniform(n-n_early,T_MIN,T_MAX)))
    order=torch.randperm(n,device=device)
    return torch.cat(xs)[order],torch.cat(ys)[order],torch.cat(zs)[order],t[order]


def global_gas_balance_loss(model, device, n=512):
    # 封閉邊界積分式氣相體積平衡的額外軟約束。
    # 空間均勻 MC 採樣，且同一組位置比較 t=0 與 t=t_sample。
    x=torch.rand(n,1,device=device)*(X_MAX-X_MIN)+X_MIN
    y=torch.rand(n,1,device=device)*(Y_MAX-Y_MIN)+Y_MIN
    z=torch.rand(n,1,device=device)*(Z_MAX-Z_MIN)+Z_MIN
    days=torch.rand(1,device=device)*3650.0
    t=torch.ones_like(x)*days*86400.0
    _,sg_t=model_physical(model,x,y,z,t)
    _,sg_0=model_physical(model,x,y,z,torch.zeros_like(t))
    reservoir_volume=(X_MAX-X_MIN)*(Y_MAX-Y_MIN)*(Z_MAX-Z_MIN)
    q_total=sum(w[3]/CO2_DENSITY_SC for w in WELLS)
    # normalized by total 10-year injected pore-volume fraction (dimensionless)
    scale=max(q_total*T_MAX/(POROSITY*reservoir_volume),1e-6)
    expected_mean=q_total*t[0,0]/(POROSITY*reservoir_volume)
    return ((POROSITY*(sg_t-sg_0).mean()/POROSITY-expected_mean)/scale)**2


def boundary_loss(model,n,device):
    # 各面抽樣，分別強制水與氣相法向流速為零
    n_face=max(8,n//6)
    losses=[]
    for axis,face in ((0,X_MIN),(0,X_MAX),(1,Y_MIN),(1,Y_MAX),(2,Z_MIN),(2,Z_MAX)):
        x=torch.rand(n_face,1,device=device)*(X_MAX-X_MIN)+X_MIN
        y=torch.rand(n_face,1,device=device)*(Y_MAX-Y_MIN)+Y_MIN
        z=torch.rand(n_face,1,device=device)*(Z_MAX-Z_MIN)+Z_MIN
        t=torch.rand(n_face,1,device=device)*T_MAX
        coords=[x,y,z]
        coords[axis]=torch.full((n_face,1),float(face),device=device,requires_grad=True)
        _,_,uw,ug=phase_fluxes(model,*coords,t)
        losses.append(torch.mean((uw[axis]/FLOW_SCALE)**2+(ug[axis]/FLOW_SCALE)**2))
    return torch.stack(losses).mean()

# -------------------------------------------------------------------------
# 7. 模型載入 / 建立
# -------------------------------------------------------------------------
@st.cache_resource
def get_model(device: str):
    model = PINNSolver()
    model.to(device)
    if MODEL_PATH.exists():
        st.success(f"已載入預訓練模型 (v5.2): {MODEL_PATH}")
        state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        model.load_state_dict(state)
    else:
        st.info("尚未找到 v5.2 模型，請先訓練。")
    return model

# -------------------------------------------------------------------------
# 8. 訓練流程（兩階段：Stage 1 只學 IC；Stage 2 加 PDE, 並強化 PDE Loss）
# -------------------------------------------------------------------------
def train_model(
    model,
    epochs_ic=100,
    epochs_full=400,
    N_pde=2048,
    N_ic=512,
    device="cpu",
):

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    status_placeholder = st.empty()
    progress_bar = st.progress(0.0)
    loss_history = {"Total Loss": [], "IC Loss": [], "PDE Loss": []}
    loss_chart = st.empty()
    st.caption("v5.2：提高初始 CO2=0 權重，並增加前 365 天 PDE 採樣；結果仍須質量守恆驗證。")

    total_epochs = epochs_ic + epochs_full
    current_step = 0

    # ------------------------- Stage 1: 只學初始條件 -------------------------
    st.write("Stage 1：訓練初始條件 (t=0, Sg=0, p=靜水壓, 無量綱壓力)")

    model.train()
    for epoch in range(epochs_ic):
        optimizer.zero_grad()

        # 只取 t=0 的點
        t_ic = torch.zeros(N_ic, 1, device=device)
        x_ic = torch.rand(N_ic, 1, device=device) * (X_MAX - X_MIN) + X_MIN
        y_ic = torch.rand(N_ic, 1, device=device) * (Y_MAX - Y_MIN) + Y_MIN
        z_ic = torch.rand(N_ic, 1, device=device) * (Z_MAX - Z_MIN) + Z_MIN

        # 正規化
        x_ic_norm = (x_ic - DOMAIN_MEANS_TENSOR[0]) / DOMAIN_SCALES_TENSOR[0]
        y_ic_norm = (y_ic - DOMAIN_MEANS_TENSOR[1]) / DOMAIN_SCALES_TENSOR[1]
        z_ic_norm = (z_ic - DOMAIN_MEANS_TENSOR[2]) / DOMAIN_SCALES_TENSOR[2]
        t_ic_norm = (t_ic - DOMAIN_MEANS_TENSOR[3]) / DOMAIN_SCALES_TENSOR[3]

        p_hat_ic_pred, Sg_ic_pred = model(x_ic_norm, y_ic_norm, z_ic_norm, t_ic_norm)

        # 物理真值 (Pa)：靜水壓
        p_ic_true = WATER_DENSITY * GRAVITY * (-z_ic)  # z<0 → -z>0 → 正壓
        # 無量綱真值
        p_ic_true_norm = p_ic_true / P_REF

        # IC Loss (在無量綱空間計)
        loss_ic_p = torch.mean((p_hat_ic_pred - p_ic_true_norm) ** 2)
        Sg_ic_true = torch.zeros_like(Sg_ic_pred)
        loss_ic_sg = torch.mean((Sg_ic_pred - Sg_ic_true) ** 2)

        # 壓力、飽和度的權重（可調）：希望壓力 profile 先穩定
        w_p_ic = 5.0
        w_s_ic = 5.0
        loss_ic = w_p_ic * loss_ic_p + w_s_ic * loss_ic_sg
        loss_pde = torch.tensor(0.0, device=device)

        total_loss = loss_ic  # Stage 1 不考慮 PDE

        total_loss.backward()
        optimizer.step()

        current_step += 1
        if epoch % 10 == 0 or epoch == epochs_ic - 1:
            status_text = (
                f"[Stage 1 - Epoch {epoch+1}/{epochs_ic}] "
                f"Total: {total_loss.item():.4e} | "
                f"IC: {loss_ic.item():.4e}"
            )
            status_placeholder.text(status_text)
            progress_bar.progress(current_step / total_epochs)
            loss_history["Total Loss"].append(total_loss.item())
            loss_history["IC Loss"].append(loss_ic.item())
            loss_history["PDE Loss"].append(loss_pde.item())
            loss_chart.line_chart(loss_history)
            print(status_text)

    # ------------------------- Stage 2: PDE + 封閉邊界 -------------------------
    st.write("Stage 2：弱可壓縮 PDE + 六面封閉邊界 + 井附近混合採樣")
    model.train()
    for epoch in range(epochs_full):
        optimizer.zero_grad()
        x_pde,y_pde,z_pde,t_pde=sample_pde(N_pde,device)
        eq_p,eq_g=compute_residuals(model,x_pde,y_pde,z_pde,t_pde)
        loss_pde_p=torch.mean(eq_p**2)
        loss_pde_sg=torch.mean(eq_g**2)
        loss_pde=loss_pde_p+loss_pde_sg
        loss_bc=boundary_loss(model,min(192,max(48,N_pde//8)),device)
        loss_mass=global_gas_balance_loss(model,device,n=384)
        t_ic=torch.zeros(N_ic,1,device=device)
        x_ic=torch.rand(N_ic,1,device=device)*(X_MAX-X_MIN)+X_MIN
        y_ic=torch.rand(N_ic,1,device=device)*(Y_MAX-Y_MIN)+Y_MIN
        z_ic=torch.rand(N_ic,1,device=device)*(Z_MAX-Z_MIN)+Z_MIN
        p_hat_ic,sg_ic=model_physical(model,x_ic,y_ic,z_ic,t_ic)
        p_ic=WATER_DENSITY*GRAVITY*(-z_ic)/P_REF
        loss_ic=5*torch.mean((p_hat_ic-p_ic)**2)+10*torch.mean(sg_ic**2)
        # PDE/BC 的單位尺度已先正規化；權重仍是示範超參數
        total_loss=loss_ic+loss_pde+loss_bc+0.5*loss_mass
        if not torch.isfinite(total_loss):
            st.error(f"Epoch {epoch+1} 出現非有限 Loss，已停止訓練")
            return
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),10.0)
        optimizer.step()
        current_step+=1
        if epoch%10==0 or epoch==epochs_full-1:
            status_text=(f"[v5.2 Stage 2 {epoch+1}/{epochs_full}] "
                         f"Total={total_loss.item():.3e} "
                         f"PDE_p={loss_pde_p.item():.3e} "
                         f"PDE_g={loss_pde_sg.item():.3e} "
                         f"BC={loss_bc.item():.3e} IC={loss_ic.item():.3e} MASS={loss_mass.item():.3e}")
            status_placeholder.text(status_text)
            progress_bar.progress(current_step/total_epochs)
            loss_history["Total Loss"].append(total_loss.item())
            loss_history["IC Loss"].append(loss_ic.item())
            loss_history["PDE Loss"].append(loss_pde.item())
            loss_chart.line_chart(loss_history)
            print(status_text)

    # ========= 4) 存成 CPU 權重 =========
    model_cpu = model.to("cpu")
    # 獨立 v5.2 權重；不覆寫 v5 的訓練成果。
    torch.save(model_cpu.state_dict(), MODEL_PATH)
    status_placeholder.success(f"訓練完成！模型已儲存至 {MODEL_PATH}")

# -------------------------------------------------------------------------
# 9. 視覺化函式
# -------------------------------------------------------------------------
def create_plots(model, time_days, n_points, z_slice, device="cpu"):
    model.to(device)
    model.eval()

    time_sec = float(time_days) * 86400.0

    # =========================================================
    # (A) 左圖：X-Y 平面（固定 z = z_slice）-> 壓力
    # =========================================================
    x = np.linspace(X_MIN, X_MAX, n_points)
    y = np.linspace(Y_MIN, Y_MAX, n_points)
    grid_x, grid_y = np.meshgrid(x, y)

    x_flat = torch.tensor(grid_x.flatten(), dtype=torch.float32, device=device).view(-1, 1)
    y_flat = torch.tensor(grid_y.flatten(), dtype=torch.float32, device=device).view(-1, 1)
    z_flat = torch.full_like(x_flat, float(z_slice))          # 固定 z = z_slice
    t_flat = torch.full_like(x_flat, float(time_sec))         # 固定時間

    # normalize
    x_norm = (x_flat - DOMAIN_MEANS_TENSOR[0]) / DOMAIN_SCALES_TENSOR[0]
    y_norm = (y_flat - DOMAIN_MEANS_TENSOR[1]) / DOMAIN_SCALES_TENSOR[1]
    z_norm = (z_flat - DOMAIN_MEANS_TENSOR[2]) / DOMAIN_SCALES_TENSOR[2]
    t_norm = (t_flat - DOMAIN_MEANS_TENSOR[3]) / DOMAIN_SCALES_TENSOR[3]

    with torch.no_grad():
        p_hat_xy, Sg_xy = model(x_norm, y_norm, z_norm, t_norm)

    print(
        f">>> [XY] z={float(z_slice):.1f}, "
        f"Sg min={Sg_xy.min().item():.6e}, "
        f"Sg max={Sg_xy.max().item():.6e}, "
        f"std={Sg_xy.std().item():.6e}"
    )

    p_xy = (p_hat_xy * P_REF).cpu().numpy().reshape(grid_x.shape)

    fig_p = go.Figure(
        data=go.Contour(
            z=p_xy / 1e6,  # Pa -> MPa
            x=x, y=y,
            colorscale="Viridis",
            colorbar_title="壓力 (MPa)",
        )
    )
    fig_p.update_layout(
        title=f"壓力分佈 (X-Y) @ {time_days:.0f} 天 (z={float(z_slice):.1f} m)",
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
    )

    # 左圖井標記 (X-Y)
    WELL_RADIUS = 300.0
    theta = np.linspace(0, 2*np.pi, 200)

    for i, (wx, wy, wz, rate) in enumerate(WELLS, start=1):
        label = f"W{i}"
        circle_x = wx + WELL_RADIUS * np.cos(theta)
        circle_y = wy + WELL_RADIUS * np.sin(theta)

        fig_p.add_trace(go.Scatter(
            x=circle_x, y=circle_y,
            mode="lines",
            line=dict(color="black", dash="dash"),
            showlegend=False
        ))
        fig_p.add_trace(go.Scatter(
            x=[wx], y=[wy],
            mode="markers+text",
            text=[label],
            textposition="top center",
            marker=dict(size=10, color="black"),
            showlegend=False
        ))

    # =========================================================
    # (B) 右圖：X-Z 剖面（固定 y = 0）-> 飽和度（越深在下面）
    # =========================================================
    x2 = np.linspace(X_MIN, X_MAX, n_points)
    z2 = np.linspace(Z_MIN, Z_MAX, n_points)
    grid_x2, grid_z2 = np.meshgrid(x2, z2)

    x2_flat = torch.tensor(grid_x2.flatten(), dtype=torch.float32, device=device).view(-1, 1)
    z2_flat = torch.tensor(grid_z2.flatten(), dtype=torch.float32, device=device).view(-1, 1)
    y2_flat = torch.zeros_like(x2_flat)                        # 固定 y=0
    t2_flat = torch.full_like(x2_flat, float(time_sec))        # 固定時間

    # normalize
    x2_norm = (x2_flat - DOMAIN_MEANS_TENSOR[0]) / DOMAIN_SCALES_TENSOR[0]
    y2_norm = (y2_flat - DOMAIN_MEANS_TENSOR[1]) / DOMAIN_SCALES_TENSOR[1]
    z2_norm = (z2_flat - DOMAIN_MEANS_TENSOR[2]) / DOMAIN_SCALES_TENSOR[2]
    t2_norm = (t2_flat - DOMAIN_MEANS_TENSOR[3]) / DOMAIN_SCALES_TENSOR[3]

    with torch.no_grad():
        _, Sg_xz = model(x2_norm, y2_norm, z2_norm, t2_norm)

    print(
        f">>> [XZ] y=0, "
        f"Sg min={Sg_xz.min().item():.6e}, "
        f"Sg max={Sg_xz.max().item():.6e}, "
        f"std={Sg_xz.std().item():.6e}"
    )

    Sg_xz_np = Sg_xz.detach().cpu().numpy().reshape(grid_x2.shape)
    Sg_xz_np = np.clip(Sg_xz_np, 0.0, 1.0)
    
    # === 動態 zmax（讓色階跟著你的數值範圍跑）===
    zmax_auto = float(np.percentile(Sg_xz_np, 99) * 1.2)
    zmax_auto = max(zmax_auto, 1e-8)
    
    print(
        "Sg_xz min/max:",
        Sg_xz_np.min(),
        Sg_xz_np.max(),
        "zmax_auto:",
        zmax_auto
    )
    
    fig_Sg = go.Figure(
        data=go.Contour(
            z=Sg_xz_np,
            x=x2,
            y=z2,
            colorscale="Plasma",
            zmin=0.0,
            zmax=zmax_auto,
            colorbar_title="CO2 飽和度",
        )
    )

    fig_Sg.update_layout(
        title=f"CO2 飽和度剖面 (X-Z) @ {time_days:.0f} 天 (y=0 m)",
        xaxis_title="X (m)",
        yaxis_title="Z (m)",
    )

        # 讓越深在下面（z 越小越深）
    fig_Sg.update_yaxes(autorange="reversed")
    
    # ====== 飽和度圖：雙井虛線（橢圓版，風格同壓力圖） ======
    WELL_RADIUS_X = 300.0          # X方向半徑（跟壓力圖一致）
    ELLIPSE_Z_RATIO = 0.35         # Z方向壓扁倍率（0.2~0.6 可調）
    WELL_RADIUS_Z = WELL_RADIUS_X * ELLIPSE_Z_RATIO
    
    theta = np.linspace(0, 2*np.pi, 200)
    
    for i, (wx, wy, wz, rate) in enumerate(WELLS, start=1):
        label = f"W{i}"
    
        # 橢圓：x 半徑 = WELL_RADIUS_X, z 半徑 = WELL_RADIUS_Z
        circle_x = wx + WELL_RADIUS_X * np.cos(theta)
        circle_z = wz + WELL_RADIUS_Z * np.sin(theta)
    
        fig_Sg.add_trace(go.Scatter(
            x=circle_x,
            y=circle_z,
            mode="lines",
            line=dict(color="black", dash="dash"),  # 跟壓力圖一致
            showlegend=False,
            hoverinfo="skip",
        ))
    
        fig_Sg.add_trace(go.Scatter(
            x=[wx],
            y=[wz],
            mode="markers+text",
            text=[label],
            textposition="top center",
            marker=dict(size=10, color="black"),     # 跟壓力圖一致
            showlegend=False,
            hoverinfo="skip",
        ))

    fig_Sg.update_xaxes(range=[X_MIN, X_MAX])
    fig_Sg.update_yaxes(range=[Z_MAX, Z_MIN])  # reversed


    return fig_p, fig_Sg

# -------------------------------------------------------------------------
# 10. 主程式流程
# -------------------------------------------------------------------------
pinn_model = get_model(DEVICE)

if st.sidebar.button("開始訓練模型 (PINN v5.2)"):
    with st.spinner("模型訓練中，視裝置與設定可能需一段時間..."):
        train_model(
            pinn_model,
            epochs_ic=int(epochs_ic),
            epochs_full=int(epochs_full),
            N_pde=int(N_PDE_SAMPLES),
            N_ic=int(N_IC_SAMPLES),
            device=DEVICE,
        )

st.header("模擬結果可視化")

col1, col2 = st.columns(2)
with st.spinner("正在生成圖表..."):
    fig_pressure, fig_saturation = create_plots(
        pinn_model, time_slider, N_POINTS_VIS, z_slice, device=DEVICE
    )

with col1:
    st.plotly_chart(fig_pressure, width="stretch")

with col2:
    st.plotly_chart(fig_saturation, width="stretch")

# v5 示範診斷：累積注入質量（不是質量守恆驗證）
st.caption("v5.2 為教學性弱可壓縮近似；尚未完成全域質量平衡及場址驗證，不可用於工程決策。")
st.caption("提醒：飽和度的色彩變化不等於已驗證的 CO2 羽流。請檢查初始條件、PDE、封閉邊界與全域質量平衡。")
st.metric("截至所選時間之理論累積注入量（噸）",f"{sum(w[3] for w in WELLS)*time_slider*86400/1000:,.0f}")


# -------------------------------------------------------------------------
# 11. 全域質量平衡診斷 + OpenUSD (USDA) 匯出
# -------------------------------------------------------------------------
RESERVOIR_VOLUME = (X_MAX-X_MIN)*(Y_MAX-Y_MIN)*(Z_MAX-Z_MIN)
INJECTION_M3_S = sum(w[3]/CO2_DENSITY_SC for w in WELLS)


def evaluate_volume_grid(model, days, resolution=24, device="cpu"):
    # 體素中心點的均勻網格積分；不使用近井偏重 PDE 樣本。
    x = np.linspace(X_MIN, X_MAX, resolution, endpoint=False)+(X_MAX-X_MIN)/(2*resolution)
    y = np.linspace(Y_MIN, Y_MAX, resolution, endpoint=False)+(Y_MAX-Y_MIN)/(2*resolution)
    z = np.linspace(Z_MIN, Z_MAX, max(8,resolution//2), endpoint=False)+(Z_MAX-Z_MIN)/(2*max(8,resolution//2))
    xx,yy,zz=np.meshgrid(x,y,z,indexing="ij")
    xyz=np.column_stack((xx.ravel(),yy.ravel(),zz.ravel())).astype(np.float32)
    sg_all=[]; p_all=[]
    model.eval()
    with torch.no_grad():
        for i in range(0,len(xyz),4096):
            coords=torch.from_numpy(xyz[i:i+4096]).to(device)
            t=torch.full((len(coords),1),float(days)*86400.0,device=device)
            ph,sg=model_physical(model,coords[:,0:1],coords[:,1:2],coords[:,2:3],t)
            sg_all.append(sg.cpu().numpy().ravel())
            p_all.append((ph*P_REF).cpu().numpy().ravel())
    return xyz,np.concatenate(sg_all),np.concatenate(p_all)


def mass_diagnostics(model, days, resolution=24, device="cpu"):
    xyz,sg,p=evaluate_volume_grid(model,days,resolution,device)
    _,sg0,p0=evaluate_volume_grid(model,0.0,resolution,device)
    cell_vol=RESERVOIR_VOLUME/len(xyz)
    injected_volume=INJECTION_M3_S*days*86400.0
    delta_gas_volume=POROSITY*float(np.sum(sg-sg0))*cell_vol
    pressure_storage=TOTAL_COMPRESSIBILITY*float(np.sum(p-p0))*cell_vol
    error_gas=delta_gas_volume-injected_volume
    error_pressure=pressure_storage-injected_volume
    return dict(xyz=xyz,sg=sg,p=p,sg0=sg0,
                injected_m3=injected_volume,gas_change_m3=delta_gas_volume,
                pressure_storage_m3=pressure_storage,
                gas_error_m3=error_gas,pressure_error_m3=error_pressure,
                gas_error_pct=(100*error_gas/injected_volume if injected_volume else None),
                pressure_error_pct=(100*error_pressure/injected_volume if injected_volume else None),
                initial_sg_mean=float(np.mean(sg0)),mean_sg=float(np.mean(sg)),
                min_sg=float(np.min(sg)),max_sg=float(np.max(sg)),
                cell_count=len(xyz))


def usd_float(x):
    return f"{float(x):.6f}"


def build_usda(data, days, stride=2, threshold=0.0):
    # USDA 是可讀文字版 OpenUSD，不需要安裝 pxr 才能產生。
    # 使用 UsdGeomPoints 的每點顏色；顏色只表示模型輸出，非經驗羽流。
    xyz=data['xyz'][::stride]
    sg=data['sg'][::stride]
    mask=sg>=threshold
    xyz=xyz[mask]; sg=sg[mask]
    # 避免檔案過大，最多 15000 點
    if len(xyz)>15000:
        step=int(np.ceil(len(xyz)/15000))
        xyz=xyz[::step];sg=sg[::step]
    lines=['#usda 1.0','(', '    defaultPrim = "World"', '    metersPerUnit = 1', '    upAxis = "Z"', ')', '', 'def Xform "World"', '{',
           '    def Cube "Reservoir"', '    {', '        double size = 1',
           '        double3 xformOp:scale = (3000, 3000, 400)',
           '        double3 xformOp:translate = (0, 0, -1800)',
           '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]',
           '        color3f[] primvars:displayColor = [(0.14, 0.28, 0.35)]',
           '        float[] primvars:displayOpacity = [0.08]', '    }']
    for idx,(wx,wy,wz,rate) in enumerate(WELLS,1):
        lines += [f'    def Cylinder "InjectionWell_{idx}"', '    {',
                  '        double radius = 12', '        double height = 400',
                  f'        double3 xformOp:translate = ({usd_float(wx)}, {usd_float(wy)}, -1800)',
                  '        uniform token[] xformOpOrder = ["xformOp:translate"]',
                  '        color3f[] primvars:displayColor = [(0.95, 0.72, 0.1)]','    }']
    lines += ['    def Points "PINN_SaturationSamples"', '    {',
              f'        custom string modelVersion = "PINN v5.2"',
              f'        custom double timeDays = {usd_float(days)}',
              '        point3f[] points = [']
    lines += [f'            ({usd_float(a)}, {usd_float(b)}, {usd_float(c)}),'
              for a,b,c in xyz]
    lines += ['        ]','        float[] widths = [']
    lines += ['            26,' for _ in xyz]
    lines += ['        ]','        color3f[] primvars:displayColor = [']
    # 0: blue, 1: orange; fixed absolute color scale 0..1
    lines += [f'            ({usd_float(0.10+0.85*v)}, {usd_float(0.25+0.30*v)}, {usd_float(0.95-0.80*v)}),'
              for v in sg]
    lines += ['        ]','        uniform token primvars:displayColor:interpolation = "vertex"',
              '        float[] primvars:Sg = [']
    lines += [f'            {usd_float(v)},' for v in sg]
    lines += ['        ]','        uniform token primvars:Sg:interpolation = "vertex"',
              '    }','}']
    return '\n'.join(lines)+'\n',len(xyz)


st.divider()
st.subheader("PINN v5.2｜質量平衡診斷與 Omniverse USD 匯出")
st.caption("以均勻體素中點積分估算全域氣相累積與壓力儲存。封閉邊界仍須另外驗證；下列差額不是嚴格的數值守恆證明。")
with st.expander("診斷與 USD 匯出設定", expanded=True):
    diag_resolution=st.select_slider("3D 積分網格（X/Y 方向）",options=[12,16,24,32],value=24)
    usd_stride=st.select_slider("USD 點雲抽樣間隔（越大檔案越小）",options=[1,2,4,8],value=2)
    usd_threshold=st.slider("USD 顯示飽和度下限（0 = 顯示所有點）",0.0,1.0,0.0,0.01)
    run_diagnostics=st.button("計算質量平衡並產生 USD 場景")

if run_diagnostics:
    with st.spinner("計算三維網格與積分中，請稍候..."):
        try:
            diagnostics=mass_diagnostics(pinn_model,time_slider,int(diag_resolution),DEVICE)
            usda_text,n_usd=build_usda(diagnostics,time_slider,int(usd_stride),usd_threshold)
            st.session_state['v52_diag']= {k:v for k,v in diagnostics.items() if k not in ('xyz','sg','p','sg0')}
            st.session_state['v52_usda']=usda_text
            st.session_state['v52_usd_count']=n_usd
            st.session_state['v52_usd_days']=time_slider
        except Exception as exc:
            st.error(f"診斷或匯出失敗：{exc}")

if 'v52_diag' in st.session_state:
    d=st.session_state['v52_diag']
    a,b,c=st.columns(3)
    a.metric("理論累積注入體積 (m³)",f"{d['injected_m3']:,.1f}")
    b.metric("模型氣相體積變化 (m³)",f"{d['gas_change_m3']:,.1f}")
    c.metric("壓力儲存體積變化 (m³)",f"{d['pressure_storage_m3']:,.1f}")
    st.write(f"氣相平衡差額：{d['gas_error_m3']:,.1f} m³；"
             f"相對注入量：{d['gas_error_pct']:.1f}%" if d['gas_error_pct'] is not None else "t=0 無注入量，無法計算相對誤差")
    if d['pressure_error_pct'] is not None:
        st.write(f"壓力儲存平衡差額：{d['pressure_error_m3']:,.1f} m³；相對注入量：{d['pressure_error_pct']:.1f}%")
    st.write(f"均勻網格 {d['cell_count']:,} 點；平均 Sg={d['mean_sg']:.5f}；t=0 平均 Sg={d['initial_sg_mean']:.5f}")
    st.write(f"Sg 最小值：{d['min_sg']:.10f}；Sg 最大值：{d['max_sg']:.10f}；差異：{d['max_sg']-d['min_sg']:.10e}")
    st.warning("診斷使用定密度氣相體積近似，且未積分邊界通量隨時間的變化。只有在邊界通量接近零、物性假設成立時，才可把差額視為近似守恆誤差。")
    st.download_button("下載 Omniverse OpenUSD 場景 (.usda)",
                       data=st.session_state['v52_usda'].encode('utf-8'),
                       file_name=f"co2_pinn_v52_day_{int(st.session_state['v52_usd_days'])}.usda",
                       mime="text/plain")
    st.caption(f"USD 含儲層示意方塊、兩口井與 {st.session_state['v52_usd_count']:,} 個模型飽和度採樣點；不是已驗證的 CO₂ 羽流。")
