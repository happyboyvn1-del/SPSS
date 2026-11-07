import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import plotly.express as px
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
from io import BytesIO

st.set_page_config(page_title="HÀO SPSS Advanced", page_icon="📊", layout="wide")

st.markdown("""
<style>
.main {background:#f6f8fc;}
.block-container {padding-top:1.5rem; padding-bottom:2rem;}
[data-testid="stMetric"] {background:white; padding:16px; border:1px solid #e8edf5; border-radius:14px;}
section[data-testid="stSidebar"] {background:#101828;}
section[data-testid="stSidebar"] * {color:#f8fafc;}
.hero {padding:25px 28px; border-radius:18px; color:white;
background:linear-gradient(115deg,#172554 0%,#2563eb 60%,#06b6d4 100%); margin-bottom:18px;}
.small-muted {color:#667085; font-size:0.9rem;}
</style>
""", unsafe_allow_html=True)

def sample_data():
    rng = np.random.default_rng(42)
    n = 160
    satisfaction = rng.normal(3.7, 0.7, n).clip(1, 5)
    quality = (satisfaction + rng.normal(0, .55, n)).clip(1, 5)
    service = (satisfaction + rng.normal(0, .65, n)).clip(1, 5)
    price = rng.normal(3.4, .8, n).clip(1, 5)
    loyalty = (0.35*quality + 0.3*service + 0.2*price + rng.normal(0, .45, n)).clip(1, 5)
    return pd.DataFrame({
        "Hai_long": quality.round(1), "Dich_vu": service.round(1),
        "Gia_ca": price.round(1), "Hai_long_chung": satisfaction.round(1),
        "Y_dinh_quay_lai": loyalty.round(1)
    })

def numeric_df(df):
    return df.select_dtypes(include=np.number).copy()

def cronbach_alpha(data):
    x = data.dropna()
    k = x.shape[1]
    if k < 2 or len(x) < 2:
        return np.nan
    variances = x.var(axis=0, ddof=1)
    total_var = x.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return (k/(k-1)) * (1 - variances.sum()/total_var)

def safe_numeric_selection(df, key, label):
    cols = list(numeric_df(df).columns)
    return st.multiselect(label, cols, default=cols[:min(4, len(cols))], key=key)

if "data" not in st.session_state:
    st.session_state.data = sample_data()

with st.sidebar:
    st.markdown("## 📊 HÀO SPSS")
    st.caption("ADVANCED DATA ANALYSIS")
    page = st.radio("ĐIỀU HƯỚNG", [
        "Tổng quan", "Dữ liệu", "Thống kê mô tả", "Cronbach's Alpha",
        "Tương quan", "Hồi quy tuyến tính", "T-test & ANOVA", "EFA"
    ])
    st.divider()
    st.caption("Phiên bản học tập • 1.0")
    st.caption("Dữ liệu được xử lý trong phiên ứng dụng.")

df = st.session_state.data.copy()
st.markdown('<div class="hero"><h1 style="margin:0">HÀO SPSS Advanced</h1><p style="margin:7px 0 0 0">Phân tích dữ liệu thống kê • Nghiên cứu • Marketing • Quản trị</p></div>', unsafe_allow_html=True)

if page == "Tổng quan":
    st.subheader("Tổng quan dữ liệu")
    a,b,c,d = st.columns(4)
    a.metric("Số quan sát", f"{len(df):,}")
    b.metric("Tổng số biến", f"{df.shape[1]:,}")
    c.metric("Biến định lượng", f"{len(numeric_df(df).columns):,}")
    d.metric("Ô dữ liệu thiếu", f"{int(df.isna().sum().sum()):,}")
    st.markdown("### Xem nhanh dữ liệu")
    st.dataframe(df.head(12), use_container_width=True)
    st.markdown("### Thống kê nhanh")
    if len(numeric_df(df).columns):
        st.dataframe(numeric_df(df).describe().T, use_container_width=True)
    st.info("Bắt đầu bằng mục **Dữ liệu** để tải file CSV/Excel hoặc chỉnh sửa bộ dữ liệu mẫu. Sau đó chọn từng công cụ ở thanh bên trái.")

elif page == "Dữ liệu":
    st.subheader("Quản lý dữ liệu")
    uploaded = st.file_uploader("Tải dữ liệu CSV hoặc Excel", type=["csv","xlsx"])
    if uploaded is not None:
        try:
            new_df = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
            st.session_state.data = new_df
            df = new_df.copy()
            st.success(f"Đã nạp {len(df)} dòng và {len(df.columns)} cột.")
        except Exception as e:
            st.error(f"Không đọc được tệp: {e}")
    c1,c2 = st.columns([1,1])
    with c1:
        if st.button("Dùng dữ liệu mẫu", use_container_width=True):
            st.session_state.data = sample_data()
            st.rerun()
    with c2:
        csv_bytes = st.session_state.data.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Tải dữ liệu CSV", csv_bytes, "hao_spss_data.csv", "text/csv", use_container_width=True)
    st.caption("Bạn có thể sửa trực tiếp các ô trong bảng. Nhấn nút lưu để áp dụng thay đổi.")
    edited = st.data_editor(st.session_state.data, num_rows="dynamic", use_container_width=True, key="data_editor")
    if st.button("Lưu thay đổi dữ liệu", type="primary"):
        st.session_state.data = edited
        st.success("Đã cập nhật dữ liệu.")
        st.rerun()

elif page == "Thống kê mô tả":
    st.subheader("Thống kê mô tả")
    cols = safe_numeric_selection(df, "desc_cols", "Chọn biến cần phân tích")
    if cols:
        result = df[cols].describe().T
        result["Median"] = df[cols].median()
        result["Mode"] = df[cols].mode().iloc[0]
        result["Missing"] = df[cols].isna().sum()
        st.dataframe(result.round(4), use_container_width=True)
        col = st.selectbox("Chọn biến để xem phân phối", cols)
        fig = px.histogram(df, x=col, marginal="box", title=f"Phân phối của {col}")
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("Xuất bảng thống kê CSV", result.to_csv().encode("utf-8-sig"), "thong_ke_mo_ta.csv")
    else:
        st.warning("Cần ít nhất một biến dạng số.")

elif page == "Cronbach's Alpha":
    st.subheader("Kiểm định độ tin cậy thang đo")
    st.write("Chọn các câu hỏi/biến quan sát thuộc cùng một thang đo.")
    cols = safe_numeric_selection(df, "alpha_cols", "Chọn các biến quan sát")
    if len(cols) >= 2:
        x = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
        alpha = cronbach_alpha(x)
        st.metric("Cronbach's Alpha", "Không tính được" if pd.isna(alpha) else f"{alpha:.3f}")
        if pd.notna(alpha):
            if alpha >= .8: st.success("Độ tin cậy nội tại tốt theo ngưỡng tham khảo phổ biến.")
            elif alpha >= .7: st.info("Độ tin cậy có thể chấp nhận trong nhiều nghiên cứu khám phá.")
            else: st.warning("Alpha thấp; cần xem xét nội dung câu hỏi và tương quan biến-tổng.")
        rows=[]
        for col in cols:
            rest=[c for c in cols if c != col]
            total_rest=x[rest].sum(axis=1)
            corr=x[col].corr(total_rest) if len(rest) else np.nan
            alpha_deleted=cronbach_alpha(x[rest]) if len(rest)>=2 else np.nan
            rows.append({"Biến":col,"Tương quan biến-tổng":corr,"Alpha nếu loại biến":alpha_deleted})
        st.dataframe(pd.DataFrame(rows).round(4), use_container_width=True)
        st.caption("Lưu ý: ngưỡng 0,7/0,3 chỉ là quy tắc tham khảo, không thay thế đánh giá lý thuyết và nội dung thang đo.")
    else:
        st.warning("Chọn tối thiểu 2 biến số.")

elif page == "Tương quan":
    st.subheader("Phân tích tương quan Pearson")
    cols = safe_numeric_selection(df, "corr_cols", "Chọn các biến")
    if len(cols) >= 2:
        method = st.selectbox("Phương pháp", ["Pearson", "Spearman"])
        clean = df[cols].dropna()
        corr = clean.corr(method=method.lower())
        st.dataframe(corr.round(3), use_container_width=True)
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title=f"Ma trận tương quan {method}")
        st.plotly_chart(fig, use_container_width=True)
        pairs=[]
        for i in range(len(cols)):
            for j in range(i+1,len(cols)):
                r,p = (stats.pearsonr(clean[cols[i]],clean[cols[j]]) if method=="Pearson"
                       else stats.spearmanr(clean[cols[i]],clean[cols[j]]))
                pairs.append({"Cặp biến":f"{cols[i]} – {cols[j]}","Hệ số":r,"p-value":p,"N":len(clean)})
        st.dataframe(pd.DataFrame(pairs).round(4), use_container_width=True)
    else:
        st.warning("Chọn ít nhất 2 biến số.")

elif page == "Hồi quy tuyến tính":
    st.subheader("Hồi quy tuyến tính OLS")
    nums = list(numeric_df(df).columns)
    if len(nums) >= 2:
        y = st.selectbox("Biến phụ thuộc (Y)", nums, index=len(nums)-1)
        xs = st.multiselect("Biến độc lập (X)", [c for c in nums if c != y], default=[c for c in nums if c != y][:2])
        if xs and st.button("Chạy hồi quy", type="primary"):
            model_df=df[[y]+xs].dropna()
            X=sm.add_constant(model_df[xs])
            model=sm.OLS(model_df[y],X).fit()
            st.metric("R-squared", f"{model.rsquared:.3f}")
            st.metric("Adjusted R-squared", f"{model.rsquared_adj:.3f}")
            st.metric("Số quan sát", len(model_df))
            st.text(model.summary().as_text())
            coef=pd.DataFrame({"Hệ số":model.params,"Sai số chuẩn":model.bse,"t":model.tvalues,"p-value":model.pvalues})
            st.dataframe(coef.round(4), use_container_width=True)
            st.caption("Kiểm tra thêm giả định mô hình, đa cộng tuyến, ngoại lệ và tính phù hợp của dữ liệu trước khi kết luận.")
    else:
        st.warning("Cần ít nhất 2 biến số.")

elif page == "T-test & ANOVA":
    st.subheader("So sánh trung bình")
    nums=list(numeric_df(df).columns)
    if nums:
        test=st.radio("Chọn kiểm định",["Independent samples T-test","Paired samples T-test","One-way ANOVA"],horizontal=False)
        value=st.selectbox("Biến định lượng",nums)
        if test=="Independent samples T-test":
            groups=[c for c in df.columns if c not in nums or df[c].nunique(dropna=True)<=10]
            group=st.selectbox("Biến phân nhóm",groups) if groups else None
            if group:
                labels=list(df[group].dropna().unique())
                if len(labels)>=2:
                    g1=st.selectbox("Nhóm 1",labels,index=0)
                    g2=st.selectbox("Nhóm 2",[v for v in labels if v!=g1])
                    x1=pd.to_numeric(df.loc[df[group]==g1,value],errors="coerce").dropna()
                    x2=pd.to_numeric(df.loc[df[group]==g2,value],errors="coerce").dropna()
                    if st.button("Chạy T-test"):
                        t,p=stats.ttest_ind(x1,x2,equal_var=False)
                        st.write({"Trung bình nhóm 1":x1.mean(),"Trung bình nhóm 2":x2.mean(),"t":t,"p-value":p,"N1":len(x1),"N2":len(x2)})
                        st.caption("Đang dùng Welch's t-test, không giả định phương sai hai nhóm bằng nhau.")
                else: st.warning("Biến phân nhóm cần có ít nhất 2 nhóm.")
        elif test=="Paired samples T-test":
            second=st.selectbox("Biến đo lần 2",[c for c in nums if c!=value])
            paired=df[[value,second]].dropna()
            if st.button("Chạy Paired T-test"):
                t,p=stats.ttest_rel(paired[value],paired[second])
                st.write({"Trung bình lần 1":paired[value].mean(),"Trung bình lần 2":paired[second].mean(),"t":t,"p-value":p,"N":len(paired)})
        else:
            groups=[c for c in df.columns if c != value and df[c].nunique(dropna=True)<=20]
            group=st.selectbox("Biến phân nhóm",groups) if groups else None
            if group:
                arrays=[pd.to_numeric(g[value],errors="coerce").dropna() for _,g in df.groupby(group)]
                arrays=[a for a in arrays if len(a)>0]
                if len(arrays)>=2 and st.button("Chạy ANOVA"):
                    f,p=stats.f_oneway(*arrays)
                    st.write({"F":f,"p-value":p,"Số nhóm":len(arrays)})
                    st.caption("ANOVA có ý nghĩa không cho biết nhóm nào khác nhau; cần kiểm định hậu nghiệm phù hợp.")
    else:
        st.warning("Không tìm thấy biến định lượng.")

elif page == "EFA":
    st.subheader("Phân tích nhân tố khám phá (EFA)")
    st.warning("EFA cần cỡ mẫu phù hợp, biến quan sát cùng thang đo và dữ liệu đáp ứng các giả định. Kết quả cần được diễn giải theo lý thuyết.")
    cols=safe_numeric_selection(df,"efa_cols","Chọn các biến quan sát")
    if len(cols)>=3:
        x=df[cols].apply(pd.to_numeric,errors="coerce").dropna()
        if len(x)>len(cols)+1:
            try:
                kmo_all,kmo_model=calculate_kmo(x)
                chi2,p=calculate_bartlett_sphericity(x)
                a,b,c=st.columns(3)
                a.metric("KMO",f"{kmo_model:.3f}")
                b.metric("Bartlett Chi-square",f"{chi2:.2f}")
                c.metric("Bartlett p-value",f"{p:.4g}")
                nf=st.number_input("Số nhân tố cần trích",min_value=1,max_value=min(len(cols),8),value=min(2,len(cols)))
                rotation=st.selectbox("Phép xoay",["varimax","promax","Không xoay"])
                if st.button("Chạy EFA",type="primary"):
                    rot=None if rotation=="Không xoay" else rotation
                    fa=FactorAnalyzer(n_factors=int(nf),rotation=rot)
                    fa.fit(x)
                    load=pd.DataFrame(fa.loadings_,index=cols,columns=[f"Factor {i+1}" for i in range(int(nf))])
                    st.markdown("#### Ma trận hệ số tải nhân tố")
                    st.dataframe(load.round(3),use_container_width=True)
                    variance=pd.DataFrame(fa.get_factor_variance(),index=["Eigenvalue","Tỷ lệ phương sai","Tỷ lệ tích lũy"],columns=[f"Factor {i+1}" for i in range(int(nf))])
                    st.markdown("#### Tổng phương sai trích")
                    st.dataframe(variance.round(3),use_container_width=True)
                    st.caption("Ứng dụng chưa tự quyết định số nhân tố tối ưu hoặc loại biến. Hãy xem scree plot, eigenvalue, hệ số tải, tải chéo và cơ sở lý thuyết.")
            except Exception as e:
                st.error(f"Không thể chạy EFA với dữ liệu hiện tại: {e}")
        else:
            st.warning("Số dòng hợp lệ chưa đủ so với số biến đã chọn.")
    else:
        st.warning("Chọn tối thiểu 3 biến quan sát dạng số.")

st.divider()
st.caption("HÀO SPSS Advanced • Công cụ hỗ trợ học tập và nghiên cứu. Luôn kiểm tra dữ liệu, giả định thống kê và diễn giải trước khi sử dụng kết quả trong báo cáo.")

