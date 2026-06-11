import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# ==========================================
# 1. 포트폴리오 최적화 함수 (국면 전환 전략 적용)
# ==========================================
def optimize_portfolio(df, target_col='Pred'):
    df = df.copy()
    df['추천비중(%)'] = 0.0
    
    # 1. 시장 국면(Good/Bad) 판단 기준 정의
    # 현재 포트폴리오에 포함된 '위험자산(주식 등)'들의 평균 AI 기댓값이 0보다 크면 좋은 장, 0 이하면 안 좋은 장
    risk_assets = df[df['자산군'] == '위험']
    market_average = risk_assets[target_col].mean() if not risk_assets.empty else df[target_col].mean()
    
    # GAPS 대회 규정 제한 조건
    MAX_INDIVIDUAL_WEIGHT = 20.0  # 종목당 최대 20% 제한
    MAX_RISK_TOTAL = 70.0        # 위험자산 총합 최대 70% 제한
    
    total_allocated = 0.0
    
    # 🟢 [좋은 시장]: 코스피, 코스닥, 반도체 등 핵심 위험자산에 자산 집중
    if market_average > 0:
        core_mask = df['ETF명'].str.contains('코스피|코스닥|반도체|200|Korea|AI', case=False, na=False)
        
        core_df = df[core_mask & (df['자산군'] == '위험')].sort_values(by=target_col, ascending=False)
        other_risk_df = df[~core_mask & (df['자산군'] == '위험')].sort_values(by=target_col, ascending=False)
        safe_df = df[df['자산군'] == '안전'].sort_values(by=target_col, ascending=False)
        
        risk_allocated = 0.0
        
        # (1순위) 코스피/코스닥/반도체 핵심 종목 
        for idx, row in core_df.iterrows():
            if risk_allocated >= MAX_RISK_TOTAL:
                break
            weight = min(MAX_INDIVIDUAL_WEIGHT, MAX_RISK_TOTAL - risk_allocated)
            df.loc[df['ETF명'] == row['ETF명'], '추천비중(%)'] = weight
            risk_allocated += weight
            
        # (2순위) 기타 위험자산
        for idx, row in other_risk_df.iterrows():
            if risk_allocated >= MAX_RISK_TOTAL:
                break
            weight = min(MAX_INDIVIDUAL_WEIGHT, MAX_RISK_TOTAL - risk_allocated)
            df.loc[df['ETF명'] == row['ETF명'], '추천비중(%)'] = weight
            risk_allocated += weight
            
        # (3순위) 남은 비중은 안전자산(채권)
        total_allocated = risk_allocated
        for idx, row in safe_df.iterrows():
            if total_allocated >= 100.0:
                break
            weight = min(MAX_INDIVIDUAL_WEIGHT, 100.0 - total_allocated)
            df.loc[df['ETF명'] == row['ETF명'], '추천비중(%)'] = weight
            total_allocated += weight
            
    # 🔴 [안 좋은 시장]: 주식 전량 매도 및 채권(안전자산) 100% 대피
    else:
        safe_df = df[df['자산군'] == '안전'].sort_values(by=target_col, ascending=False)
        
        for idx, row in safe_df.iterrows():
            if total_allocated >= 100.0:
                break
            weight = min(MAX_INDIVIDUAL_WEIGHT, 100.0 - total_allocated)
            df.loc[df['ETF명'] == row['ETF명'], '추천비중(%)'] = weight
            total_allocated += weight
            
    return df

# ==========================================
# 2. 메인 앱 레이아웃 및 탭 구성
# ==========================================
st.set_page_config(page_title="GAPS AI 포트폴리오", layout="wide")
st.title("🏆 GAPS 대회용 AI 포트폴리오 대시보드")

# 🚨 주의: 이 부분은 회원님의 기존 데이터 로딩 코드(CSV 등)로 교체해 주세요.
# 예시: df_daily = pd.read_csv("my_data.csv")
# 아래는 에러 방지용 데이터 유무 체크입니다.
if 'df_daily' not in locals():
    st.error("데이터가 로드되지 않았습니다. df_daily 변수를 선언하는 기존 코드를 위에 추가해주세요.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["오늘의 디렉션", "섹터 TOP 3", "🔥 캘린더 연동 백테스트", "현재 추세"])

# (탭 1, 2, 4는 기존 코드의 내용을 유지하시면 됩니다)
with tab1:
    st.write("오늘의 추천 포트폴리오 화면입니다. (기존 코드 유지)")

with tab2:
    st.write("섹터별 TOP 3 화면입니다. (기존 코드 유지)")

with tab4:
    st.write("현재 추세 화면입니다. (기존 코드 유지)")

# ==========================================
# 3. 백테스트 탭 (색상 그라데이션 에러 해결 & 히스토리 기능 추가)
# ==========================================
with tab3:
    st.subheader("🔥 캘린더 연동 타임머신 백테스트")
    if not df_daily.empty:
        valid_dates = sorted(df_daily['Date'].unique())
        min_date = datetime.strptime(valid_dates[0], '%Y-%m-%d').date()
        max_date = datetime.strptime(valid_dates[-1], '%Y-%m-%d').date()
        
        col1, col2 = st.columns([1, 1])
        with col1:
            target_date = st.date_input("🗓️ 백테스트 기준일 선택:", value=max_date, min_value=min_date, max_value=max_date)
        with col2:
            period = st.radio("⏳ 시뮬레이션 기간 선택:", ["1일 (1영업일)", "1주 (5영업일)", "2주 (10영업일)", "1달 (20영업일)"], horizontal=True)
        
        target_date_str = target_date.strftime('%Y-%m-%d')
        days = 1 if "1일" in period else (5 if "1주" in period else (10 if "2주" in period else 20))
        
        df_filtered = df_daily[df_daily['Date'] <= target_date_str]
        
        if df_filtered.empty or len(df_filtered['Date'].unique()) == 0:
            st.warning("선택하신 날짜에 해당하는 데이터가 부족합니다.")
        else:
            unique_dates = sorted(df_filtered['Date'].unique())[-days:]
            df_period = df_filtered[df_filtered['Date'].isin(unique_dates)]
            
            daily_model_returns, daily_market_returns, daily_max_returns, dates_list = [], [], [], []
            target_date_portfolio = None 
            portfolio_history = [] 
            
            for d in unique_dates:
                df_d = df_period[df_period['Date'] == d]
                
                past_portfolio = optimize_portfolio(df_d, target_col='Pred')
                
                # 매일매일 모델이 담았던 종목과 비중 기록
                for _, p_row in past_portfolio.iterrows():
                    if p_row['추천비중(%)'] > 0:
                        portfolio_history.append({
                            '날짜': d,
                            'ETF명': p_row['ETF명'],
                            '비중(%)': p_row['추천비중(%)']
                        })
                
                daily_port_return = float((past_portfolio['추천비중(%)'] / 100 * past_portfolio['실제수익률(%)'].fillna(0)).sum())
                daily_model_returns.append(daily_port_return / 100)
                
                if d == unique_dates[-1]:
                    target_date_portfolio = past_portfolio 
                
                daily_market_returns.append(float(df_d['Actual'].mean() / 100))
                
                max_portfolio = optimize_portfolio(df_d, target_col='Actual')
                daily_max_return = float((max_portfolio['추천비중(%)'] / 100 * max_portfolio['실제수익률(%)'].fillna(0)).sum())
                daily_max_returns.append(daily_max_return / 100)
                
                dates_list.append(d)
            
            cum_model = (np.cumprod(1 + np.array(daily_model_returns)) - 1) * 100
            cum_market = (np.cumprod(1 + np.array(daily_market_returns)) - 1) * 100
            cum_max = (np.cumprod(1 + np.array(daily_max_returns)) - 1) * 100
            
            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"🤖 모델 포트폴리오 성과", f"{cum_model[-1]:.2f}%")
            c2.metric(f"📊 시장 평균 1/N 성과", f"{cum_market[-1]:.2f}%")
            c3.metric("✨ 알파 (초과 수익)", f"{(cum_model[-1] - cum_market[-1]):.2f}%")
            c4.metric("👑 신의 영역 (이론적 최대)", f"{cum_max[-1]:.2f}%")
            
            df_chart = pd.DataFrame({
                '👑 신의 영역 (완벽한 예지력)': cum_max,
                '🤖 동적 가중치 앙상블 모델': cum_model, 
                '📊 시장 전체 평균 1/N': cum_market
            }, index=dates_list)
            st.line_chart(df_chart, use_container_width=True)

            st.markdown("### 📜 시뮬레이션 기간 내 일자별 종목 비중(%) 변화 흐름")
            if portfolio_history:
                df_hist = pd.DataFrame(portfolio_history)
                df_pivot = df_hist.pivot(index='날짜', columns='ETF명', values='비중(%)').fillna(0)
                
                # 색상 에러를 피하기 위해 .background_gradient()를 제거하고 깔끔한 숫자로 출력
                st.dataframe(df_pivot.style.format("{:.1f}%"), use_container_width=True)

            st.markdown(f"### 🔍 기준일({dates_list[-1]}) 최종 포트폴리오 상세 내역")
            if target_date_portfolio is not None:
                st.dataframe(target_date_portfolio.style.format({
                    '추천비중(%)': '{:.1f}%', 
                    '앙상블 기댓값(%)': '{:.3f}%',
                    '실제수익률(%)': '{:.3f}%'
                }), use_container_width=True)