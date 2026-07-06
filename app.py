import os

import numpy as np
import pandas as pd
import streamlit as st

from model_utils import BASE_INPUT_COLS, DISPLAY_COLS, load_artifacts, predict_tau

APP_TITLE = "Neuromorphic Retention Search Engine"
MODEL_DIR = "models"

st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")

CUSTOM_CSS = """
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
.main-card {
    border: 1px solid rgba(49, 51, 63, 0.15);
    border-radius: 18px;
    padding: 1.2rem 1.4rem;
    background: rgba(250, 250, 250, 0.7);
}
.big-title {font-size: 2.2rem; font-weight: 800; margin-bottom: 0.2rem;}
.subtitle {font-size: 1rem; color: #555; margin-bottom: 1rem;}
.badge {
    display: inline-block;
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    background: #eef3ff;
    margin-right: 0.35rem;
    font-size: 0.85rem;
    font-weight: 600;
}
.result-number {font-size: 2.2rem; font-weight: 800;}
.small-muted {color: #666; font-size: 0.9rem;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def cached_artifacts():
    return load_artifacts(MODEL_DIR)


@st.cache_data
def load_candidates():
    path = os.path.join(MODEL_DIR, "candidate_predictions.csv")
    return pd.read_csv(path)


model, schema, summary = cached_artifacts()
candidates = load_candidates()
feature_columns = schema["feature_columns"]

# Keep only useful candidate display columns if available.
CANDIDATE_DISPLAY_COLS = [
    "Channel", "Process", "polymer", "Cation", "Anion", "Ion_type",
    "Gate_voltage_V", "Drain_voltage_V", "Gate_pulse_width_ms", "Pulse_number",
    "Annealing_temp_C", "Predicted_Tau_ms", "Tau_ms"
]

st.markdown(
    f"""
    <div class="main-card">
      <div class="big-title">🧠 {APP_TITLE}</div>
      <div class="subtitle">
        Electrochemical EGST 기반 뉴로모픽 소자의 소재·공정·구동 조건을 검색하면 예상 retention time을 출력하는 발표용 웹 프로토타입입니다.
      </div>
      <span class="badge">Condition → Retention</span>
      <span class="badge">Target → Candidate</span>
      <span class="badge">Research Screening Tool</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("모델 요약")
    st.metric("Hold-out R²_log", f"{summary.get('r2_log_holdout', 0):.3f}")
    st.metric("MAE_log", f"{summary.get('mae_log_holdout', 0):.3f}")
    st.metric("학습 데이터", f"{summary.get('n_rows_after_filtering', len(candidates))} rows")
    st.divider()
    st.markdown(
        """
        **발표용 핵심 문장**  
        Retention time을 실험 후 확인하는 값에서, 실험 전 검색하고 설계할 수 있는 값으로 전환했습니다.
        """
    )
    st.caption("주의: 본 앱은 실험 결과를 확정하는 상용 엔진이 아니라 후보 조건을 줄이는 screening tool입니다.")


def categorical_input(col, label=None):
    opts = schema["categorical"].get(col, [])
    if not opts:
        return st.text_input(label or col, value="Missing")
    default_idx = 0
    preferred_by_col = {
        "Channel": ["P3HT", "PEDOT", "IGZO"],
        "Process": ["Spin-coating", "spin coating", "Spin coating"],
        "Ion_type": ["ion_gel", "Ion gel", "electrolyte"],
        "polymer": ["PVDF-HFP", "PSS", "PEO"],
        "Cation": ["BMIM", "Li", "Na"],
        "Anion": ["TFSI", "Cl", "PF6"],
        "Electrode_type": ["Au", "Pt", "ITO"],
    }
    for pref in preferred_by_col.get(col, []):
        if pref in opts:
            default_idx = opts.index(pref)
            break
    return st.selectbox(label or col, opts, index=default_idx)


def numeric_input(col, label=None, help_text=None):
    meta = schema["numeric"].get(col, {"min": 0.0, "max": 1.0, "default": 0.0})
    mn, mx, default = meta["min"], meta["max"], meta["default"]
    if not np.isfinite(mn):
        mn = 0.0
    if not np.isfinite(mx):
        mx = max(mn + 1.0, default + 1.0)
    if mx <= mn:
        mx = mn + 1.0
    if not np.isfinite(default):
        default = mn
    default = min(max(default, mn), mx)
    step = float(max((mx - mn) / 100, 0.001))
    return st.number_input(label or col, min_value=float(mn), max_value=float(mx), value=float(default), step=step, help=help_text)


def build_search_form():
    user_input = {}
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("① 소재·공정 조건")
        for col in ["Channel", "Solvent", "Concentration_mg_ml", "Process", "Spin_RPM", "Annealing_temp_C", "Annealing_time_h"]:
            if col in schema["categorical"]:
                user_input[col] = categorical_input(col)
            elif col in schema["numeric"]:
                user_input[col] = numeric_input(col)

    with col2:
        st.subheader("② 전해질·이온 조건")
        for col in ["Ion_type", "wt", "polymer", "Cation", "Anion", "Ion_diffusion", "Ion_viscosity", "Anion_radius", "Cation_radius"]:
            if col in schema["categorical"]:
                user_input[col] = categorical_input(col)
            elif col in schema["numeric"]:
                user_input[col] = numeric_input(col)

    with col3:
        st.subheader("③ 전기적 구동 조건")
        for col in ["Gate_voltage_V", "Drain_voltage_V", "Gate_pulse_width_ms", "Pulse_number", "Electrode_type", "Operating_temp_C"]:
            if col in schema["categorical"]:
                user_input[col] = categorical_input(col)
            elif col in schema["numeric"]:
                user_input[col] = numeric_input(col)

    return user_input


def add_reliability_band(pred_log):
    mae_log = float(summary.get("mae_log_holdout", 0.6))
    lower = max(0.0, np.expm1(pred_log - mae_log))
    upper = np.expm1(pred_log + mae_log)
    return lower, upper


tab1, tab2, tab3, tab4 = st.tabs([
    "🔎 조건 검색",
    "🎯 목표값 검색",
    "📚 유사 실험 검색",
    "🗣️ 발표/사업화 포인트",
])

with tab1:
    st.header("공정 조건 입력 → 예상 Retention Time 검색")
    st.write("소재, 전해질, 공정, 구동 조건을 선택한 뒤 검색 버튼을 누르면 모델이 예상 τ 값을 계산합니다.")

    with st.form("condition_search_form"):
        user_input = build_search_form()
        submitted = st.form_submit_button("🔎 Retention Time 검색", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([user_input])
        result = predict_tau(model, input_df, feature_columns)
        tau_ms = float(result["Predicted_Tau_ms"].iloc[0])
        pred_log = float(result["Predicted_log1p_Tau"].iloc[0])
        lower, upper = add_reliability_band(pred_log)

        st.success("검색 결과가 계산되었습니다.")
        a, b, c, d = st.columns(4)
        a.metric("Predicted τ", f"{tau_ms:,.2f} ms")
        b.metric("초 단위", f"{tau_ms / 1000:,.2f} s")
        c.metric("참고 범위 하한", f"{lower:,.1f} ms")
        d.metric("참고 범위 상한", f"{upper:,.1f} ms")

        st.subheader("입력 조건 요약")
        st.dataframe(input_df, use_container_width=True)

        st.info(
            "이 값은 제작 성공을 보장하는 정답이 아니라, 반복 실험 전에 우선 검토할 후보 조건을 줄이기 위한 예측값입니다."
        )

with tab2:
    st.header("원하는 Retention Time 입력 → 후보 조건 추천")
    st.write("목표 τ 범위를 입력하면 문헌 기반 후보 조건을 모델로 재평가해 목표값에 가까운 조합을 추천합니다.")

    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        target_min = st.number_input("목표 τ 최소값(ms)", min_value=0.0, value=1000.0, step=100.0)
    with col_b:
        target_max = st.number_input("목표 τ 최대값(ms)", min_value=0.0, value=100000.0, step=1000.0)
    with col_c:
        top_n = st.slider("추천 조건 개수", min_value=3, max_value=20, value=8)

    if target_max <= target_min:
        st.warning("최대값은 최소값보다 크게 설정하세요.")
    else:
        target_mid_log = np.log1p((target_min + target_max) / 2)
        temp = candidates.copy()
        temp["Predicted_Tau_ms"] = pd.to_numeric(temp["Predicted_Tau_ms"], errors="coerce")
        temp["distance_to_target"] = (np.log1p(temp["Predicted_Tau_ms"].clip(lower=0)) - target_mid_log).abs()
        temp["target_range_match"] = temp["Predicted_Tau_ms"].between(target_min, target_max)
        temp = temp.sort_values(["target_range_match", "distance_to_target"], ascending=[False, True]).head(top_n)

        cols = [c for c in CANDIDATE_DISPLAY_COLS + ["target_range_match"] if c in temp.columns]
        st.dataframe(temp[cols], use_container_width=True)
        chart_df = temp[["Predicted_Tau_ms"]].copy()
        chart_df.index = [f"Candidate {i+1}" for i in range(len(chart_df))]
        st.bar_chart(chart_df)
        st.caption("Reverse design은 후보 조건 생성 → 모델 예측 → 목표값 근접도 기준 ranking 방식입니다.")

with tab3:
    st.header("입력 조건과 가까운 문헌 기반 후보 찾기")
    st.write("발표 Q&A에서 '기존 데이터와 연결되어 있느냐'는 질문에 대비하기 위한 탭입니다.")

    simple_cols = st.columns(4)
    with simple_cols[0]:
        channel_filter = st.selectbox("Channel 필터", ["전체"] + sorted(candidates["Channel"].dropna().astype(str).unique().tolist()) if "Channel" in candidates.columns else ["전체"])
    with simple_cols[1]:
        polymer_filter = st.selectbox("Polymer 필터", ["전체"] + sorted(candidates["polymer"].dropna().astype(str).unique().tolist()) if "polymer" in candidates.columns else ["전체"])
    with simple_cols[2]:
        cation_filter = st.selectbox("Cation 필터", ["전체"] + sorted(candidates["Cation"].dropna().astype(str).unique().tolist()) if "Cation" in candidates.columns else ["전체"])
    with simple_cols[3]:
        max_rows = st.slider("표시 개수", min_value=5, max_value=50, value=15)

    filtered = candidates.copy()
    if channel_filter != "전체" and "Channel" in filtered.columns:
        filtered = filtered[filtered["Channel"].astype(str) == channel_filter]
    if polymer_filter != "전체" and "polymer" in filtered.columns:
        filtered = filtered[filtered["polymer"].astype(str) == polymer_filter]
    if cation_filter != "전체" and "Cation" in filtered.columns:
        filtered = filtered[filtered["Cation"].astype(str) == cation_filter]

    display_cols = [c for c in CANDIDATE_DISPLAY_COLS if c in filtered.columns]
    st.dataframe(filtered[display_cols].head(max_rows), use_container_width=True)

with tab4:
    st.header("발표에서 강조할 포인트")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("결과물 완성도")
        st.markdown(
            """
            - 데이터셋 구축, 전처리, 모델 학습, 성능 평가를 끝낸 뒤 웹 인터페이스로 연결했습니다.
            - 사용자는 코드가 아니라 브라우저에서 조건을 입력하고 결과를 확인합니다.
            - 예측 기능과 목표값 기반 추천 기능을 함께 제공해 단순 분석 모델보다 사용성이 높습니다.
            """
        )
        st.subheader("실현가능성")
        st.markdown(
            """
            - 현재 버전은 연구자용 screening tool입니다.
            - 상용화 시 신규 논문/실험 데이터를 누적해 모델을 주기적으로 업데이트할 수 있습니다.
            - SaaS 형태로 연구실, 소재 기업, 소자 스타트업에 제공할 수 있습니다.
            """
        )
    with col2:
        st.subheader("사회적 기여도")
        st.markdown(
            """
            - 반복 실험 감소로 시약, 용매, 장비 사용 시간을 줄일 수 있습니다.
            - 고가 장비 접근성이 낮은 연구팀도 후보 조건을 먼저 좁힐 수 있습니다.
            - 장기적으로 저전력 엣지 AI, 웨어러블 헬스케어, 인공 감각 소자 개발을 앞당길 수 있습니다.
            """
        )
        st.subheader("Q&A 방어 문장")
        st.info(
            "이 모델은 실험을 대체하는 도구가 아니라, 실험 전에 가능성이 높은 조건을 선별하는 의사결정 보조 도구입니다."
        )
