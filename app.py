import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import os
import re
from datetime import datetime, timedelta

# 웹페이지 기본 설정
st.set_page_config(layout="wide", page_title="GAPS 주도주 트렌드 대시보드")

st.title("🥇 GAPS 앙상블 대시보드 [V40 - 알고리즘 탑재형 방향성 버전]")
st.markdown("💡 **시스템 안내:** 10개년 패턴 분석 알고리즘은 그대로 작동하며, 출력값은 **코스피/코스닥/미국반도체**에 한해 **상승/하락**으로만 직관적으로 표기됩니다.")

# --- 캐시 강제 초기화 버튼 ---
st.sidebar.header("🔄 데이터 동기화")
if st.sidebar.button("🔄 최신 데이터 강제 갱신", type="primary", use_container_width=True):
    st.cache_data.clear()  
    st.sidebar.success("⏳ 캐시가 초기화되었습니다! 최신 데이터를 다시 불러옵니다...")
    st.rerun()

# --- 좌측 사이드바 설정 ---
st.sidebar.header("⚙️ 앙상블 모델 설정")
train_window_option = st.sidebar.selectbox(
    "최근 장세 반영 기간 (회귀 가중치용)",
    ["500 영업일 (약 2년)", "700 영업일 (약 2.8년)", "1000 영업일 (약 4년)"],
    index=0
)

# ==========================================
# 1. 포트폴리오 최적화 함수 (기존 수학적 연산 유지)
# ==========================================
def optimize_portfolio(df_predictions, target_col='Pred'):
    df = df_predictions.copy()
    df['추천비중(%)'] = 0.0
    
    col = target_col
    if col == 'Pred' and '앙상블 기댓값(%)' in df.columns:
        col = '앙상블 기댓값(%)'
    elif col == 'Actual' and '실제수익률(%)' in df.columns:
        col = '실제수익률(%)'
        
    if '자산군' not in df.columns:
        def get_asset_type(cat):
            cat = str(cat).replace(' ', '').replace('_', '')
            if '채권' in cat or '금리' in cat or '현금' in cat or '초단기' in cat: return '안전'
            return '위험'
        df['자산군'] = df['카테고리'].apply(get_asset_type)
        
    risk_assets = df[df['자산군'] == '위험']
    market_average = risk_assets[col].mean() if not risk_assets.empty else df[col].mean()
    
    MAX_INDIVIDUAL = 20.0  
    MAX_RISK = 70.0        
    total_allocated = 0.0
    
    if market_average > 0:
        core_df = df[df['자산군'] == '위험'].sort_values(by=col, ascending=False)
        safe_df = df[df['자산군'] == '안전'].sort_values(by=col, ascending=False)
        
        risk_allocated = 0.0
        for idx, row in core_df.iterrows():
            if risk_allocated >= MAX_RISK: break
            weight = min(MAX_INDIVIDUAL, MAX_RISK - risk_allocated)
            df.loc[idx, '추천비중(%)'] = weight
            risk_allocated += weight
            
        total_allocated = risk_allocated
        for idx, row in safe_df.iterrows():
            if total_allocated >= 100.0: break
            weight = min(MAX_INDIVIDUAL, 100.0 - total_allocated)
            df.loc[idx, '추천비중(%)'] = weight
            total_allocated += weight
    else:
        safe_df = df[df['자산군'] == '안전'].sort_values(by=col, ascending=False)
        for idx, row in safe_df.iterrows():
            if total_allocated >= 100.0: break
            weight = min(MAX_INDIVIDUAL, 100.0 - total_allocated)
            df.loc[idx, '추천비중(%)'] = weight
            total_allocated += weight
            
    if total_allocated < 100.0:
        cash_weight = 100.0 - total_allocated
        cash_row = pd.DataFrame([{
            'ETF명': '현금보유 (Cash)', '카테고리': '현금', '자산군': '안전', 
            '추천비중(%)': cash_weight, col: 0.0
        }])
        if 'Actual' in df.columns: cash_row['Actual'] = 0.0
        df = pd.concat([df, cash_row], ignore_index=True)

    return df

# ==========================================
# 2. 오리지널 데이터 로딩 및 10년 패턴 분석 (필터링 강화)
# ==========================================
@st.cache_data(ttl=21600, show_spinner="⏳ 지정 자산 패턴 연산 및 실시간 시세 파싱 중...")
def run_full_analysis(df_raw, train_window_option):
    ticker_dict = {}
    header_idx = -1
    for idx, row in df_raw.iterrows():
        row_str = " ".join(row.fillna('').astype(str))
        if '티커' in row_str or 'ETF명' in row_str:
            header_idx = idx
            break

    if header_idx != -1:
        columns_row = df_raw.iloc[header_idx].fillna('').astype(str).str.strip().str.replace('\n', '')
        df_data = df_raw.iloc[header_idx+1:].copy()
        df_data.columns = columns_row
        t_col = next((str(c) for c in df_data.columns if '티커' in str(c) or '종목' in str(c)), str(df_data.columns[0]))
        n_col = next((str(c) for c in df_data.columns if 'ETF' in str(c) or '명' in str(c)), str(df_data.columns[1]))
        c2_col = next((str(c) for c in df_data.columns if '구분2' in str(c)), str(df_data.columns[-1]))

        for _, row in df_data.iterrows():
            ticker = str(row[t_col]).strip()
            name = str(row[n_col]).strip()
            category = str(row[c2_col]).strip()
            
            # 🟢 [핵심 변경] 원천 데이터 적재 시점부터 코스피, 코스닥, 반도체 및 안전자산(채권/현금)만 필터링
            is_target_stock = bool(re.search('코스피|코스닥|반도체|필라델피아|나스닥|200|Korea|SOXX|KOSPI|KOSDAQ', name, re.IGNORECASE))
            is_safe_asset = bool(re.search('채권|금리|현금|초단기', category))
            
            if is_target_stock or is_safe_asset:
                if ticker.upper().startswith('A'): ticker = ticker[1:]
                if len(ticker) == 6 and ticker.isalnum():
                    ticker_dict[ticker] = {'name': name, 'category': category}

    summary_results = []
    daily_records = []
    start_date = (datetime.today() - timedelta(days=365*10)).strftime('%Y-%m-%d')

    if "1000" in train_window_option: fit_window = 1000
    elif "700" in train_window_option: fit_window = 700
    else: fit_window = 500

    actual_latest_date = "데이터 없음"

    for ticker, info in ticker_dict.items():
        try:
            df = fdr.DataReader(ticker, start_date)[['Close']].rename(columns={'Close': 'Price'})
            if len(df) < 40: continue

            actual_latest_date = df.index[-1].strftime('%Y-%m-%d')
            df['Price_Change'] = df['Price'].pct_change()
            df['Dir'] = np.where(df['Price_Change'] > 0, 1, 0)
            
            df['Target_Return'] = df['Price_Change']  
            df['L1'] = df['Dir'].shift(1)             
            df['L2'] = df['Dir'].shift(2)             
            df['L3'] = df['Dir'].shift(3)             
            df['L4'] = df['Dir'].shift(4)             
            
            df_clean = df.dropna(subset=['L4', 'Target_Return']).copy()
            if len(df_clean) < 100: continue
            
            annual_std_dev = df_clean['Price_Change'].std() * np.sqrt(252) * 100
            
            e1_map = df_clean.groupby('L1')['Target_Return'].mean().to_dict()
            e2_map = df_clean.groupby(['L1', 'L2'])['Target_Return'].mean().to_dict()
            e3_map = df_clean.groupby(['L1', 'L2', 'L3'])['Target_Return'].mean().to_dict()
            e4_map = df_clean.groupby(['L1', 'L2', 'L3', 'L4'])['Target_Return'].mean().to_dict()
            
            df_clean['E1'] = df_clean['L1'].map(e1_map).fillna(0.0)
            df_clean['E2'] = df_clean.set_index(['L1', 'L2']).index.map(e2_map.get).fillna(0.0)
            df_clean['E3'] = df_clean.set_index(['L1', 'L2', 'L3']).index.map(e3_map.get).fillna(0.0)
            df_clean['E4'] = df_clean.set_index(['L1', 'L2', 'L3', 'L4']).index.map(e4_map.get).fillna(0.0)
            
            actual_w = min(len(df_clean), fit_window)
            df_fit = df_clean.tail(actual_w)
            
            X_fit = df_fit[['E1', 'E2', 'E3', 'E4']].values
            y_fit = df_fit['Target_Return'].values
            X_design = np.column_stack((np.ones(len(X_fit)), X_fit))
            
            coeffs, _, _, _ = np.linalg.lstsq(X_design, y_fit, rcond=None)
            best_intercept = coeffs[0]
            best_w1, best_w2, best_w3, best_w4 = coeffs[1:]
            
            df_clean['Final_Pred'] = (best_intercept + df_clean['E1']*best_w1 + df_clean['E2']*best_w2 + df_clean['E3']*best_w3 + df_clean['E4']*best_w4) * 100
            df_clean['Actual'] = df_clean['Target_Return'] * 100
            
            final_correlation = df_clean['Final_Pred'].corr(df_clean['Actual'])
            if pd.isna(final_correlation): final_correlation = 0.0
            
            for date, row in df_clean.tail(60).iterrows():
                daily_records.append({
                    'Date': date.strftime('%Y-%m-%d'), 'ETF명': info['name'], '카테고리': info['category'],
                    'Pred': row['Final_Pred'], 'Actual': row['Actual']
                })

            next_L1 = df['Dir'].iloc[-1]
            next_L2 = df['Dir'].iloc[-2]
            next_L3 = df['Dir'].iloc[-3]
            next_L4 = df['Dir'].iloc[-4]
            
            next_E1 = e1_map.get(next_L1, 0.0)
            next_E2 = e2_map.get((next_L1, next_L2), 0.0)
            next_E3 = e3_map.get((next_L1, next_L2, next_L3), 0.0)
            next_E4 = e4_map.get((next_L1, next_L2, next_L3, next_L4), 0.0)
            
            pred_tomorrow = (best_intercept + next_E1*best_w1 + next_E2*best_w2 + next_E3*best_w3 + next_E4*best_w4) * 100

            df['MA20'] = df['Price'].rolling(20).mean()
            df['Trend'] = np.where(df['Price'] >= df['MA20'], "🟢상승세", "🔴하락세")

            summary_results.append({
                '종목코드': 'A' + ticker, 'ETF명': info['name'], '카테고리': info['category'],
                '현재추세': df['Trend'].iloc[-1], '앙상블 기댓값(%)': pred_tomorrow,
                '연간 표준편차(%)': annual_std_dev, '모델 상관계수': final_correlation
            })
        except: pass
        
    return pd.DataFrame(summary_results), pd.DataFrame(daily_records), actual_latest_date

# ==========================================
# 3. 메인 레이아웃 및 방향성 위주 화면 출력
# ==========================================
csv_filename = "gaps_etf_list.csv"

if os.path.exists(csv_filename):
    df_raw = None
    for enc in ['utf-8-sig', 'cp949', 'utf-8', 'euc-kr']:
        try:
            df_raw = pd.read_csv(csv_filename, header=None, encoding=enc, dtype=str)
            break
        except: continue

    if df_raw is not None:
        df_analysis, df_daily, latest_market_date = run_full_analysis(df_raw, train_window_option)
        
        df_pred_today = df_analysis.rename(columns={'앙상블 기댓값(%)': 'Pred'})
        optimal_portfolio_full = optimize_portfolio(df_pred_today, target_col='Pred')
        optimal_portfolio = optimal_portfolio_full[optimal_portfolio_full['추천비중(%)'] > 0].copy()

        st.sidebar.markdown("---")
        st.sidebar.info(f"📅 **최신 데이터 기준일:**\n`{latest_market_date}`")
        
        tab1, tab2, tab3 = st.tabs([
            "🎯 오늘의 추천 자산 및 예측 방향", 
            "📋 전체 대상 종목 예측 상태", 
            "💼 내일 아침 매매 디렉션"
        ])

        with tab1:
            st.subheader("🎯 오늘의 최적화 포트폴리오 비중 및 방향성")
            
            risk_sum = float(optimal_portfolio[optimal_portfolio['자산군'] == '위험']['추천비중(%)'].sum())
            safe_sum = float(optimal_portfolio[optimal_portfolio['자산군'] == '안전']['추천비중(%)'].sum())
            
            m1, m2 = st.columns(2)
            m1.metric("🔴 선정 주도주 위험자산 비중 (Max 70%)", f"{risk_sum:.1f}%")
            m2.metric("🟢 방어용 안전자산 비중", f"{safe_sum:.1f}%")
            
            # 🟢 [핵심 변경] 소수점 퍼센트 수치 대신 상승/하락 텍스트로 치환하여 노출
            display_portfolio = optimal_portfolio[['자산군', '카테고리', 'ETF명', '추천비중(%)', 'Pred']].copy()
            display_portfolio['AI 예측 방향'] = np.where(display_portfolio['Pred'] > 0, "상승 🟢", "하락 🔴")
            display_portfolio.loc[display_portfolio['ETF명'].str.contains('현금|Cash'), 'AI 예측 방향'] = "대기 🟡"
            
            st.dataframe(
                display_portfolio[['자산군', '카테고리', 'ETF명', '추천비중(%)', 'AI 예측 방향']].style.format({'추천비중(%)': '{:.1f}%'}), 
                use_container_width=True, hide_index=True
            )

        with tab2:
            st.subheader("📋 전체 필터링 대상 종목의 예측 방향 상태")
            
            # 위험 자산(지정 주도주) 리스트 분리 표기
            df_stocks = df_analysis[df_analysis['카테고리'].str.contains('지수|주식|반도체|코스피|코스닥|나스닥')].copy()
            df_stocks['AI 예측 방향'] = np.where(df_stocks['앙상블 기댓값(%)'] > 0, "상승 🟢", "하락 🔴")
            
            st.markdown("### 📈 코스피 / 코스닥 / 미국 반도체 섹터 상태")
            st.dataframe(
                df_stocks[['카테고리', 'ETF명', '현재추세', 'AI 예측 방향']], 
                use_container_width=True, hide_index=True
            )
            
            # 안전 자산 리스트 분리 표기
            df_bonds = df_analysis[~df_analysis['ETF명'].isin(df_stocks['ETF명'])].copy()
            df_bonds['AI 예측 방향'] = np.where(df_bonds['앙상블 기댓값(%)'] > 0, "상승 🟢", "하락 🔴")
            
            st.markdown("### 🛡️ 채권 및 현금성 안전자산 상태")
            st.dataframe(
                df_bonds[['카테고리', 'ETF명', '현재추세', 'AI 예측 방향']], 
                use_container_width=True, hide_index=True
            )

        with tab3:
            st.subheader("💼 내 포트폴리오 리밸런싱 가이드")
            st.markdown("보유 중인 종목 비중을 입력하면 목표 비중에 맞춘 매매 행동 지침이 출력됩니다.")

            all_etf_names = df_analysis['ETF명'].tolist() + ['현금보유 (Cash)']
            selected_holdings = st.multiselect("📌 1. 현재 계좌 보유 종목 선택:", all_etf_names)

            if selected_holdings:
                df_input = pd.DataFrame({"ETF명": selected_holdings, "현재 비중(%)": [100.0 / len(selected_holdings)] * len(selected_holdings)})
                edited_holdings = st.data_editor(df_input, use_container_width=True, hide_index=True)
                total_weight = edited_holdings["현재 비중(%)"].sum()
                
                if abs(total_weight - 100.0) > 0.1:
                    st.warning(f"⚠️ 현재 비중의 합이 {total_weight:.1f}% 입니다. 100%에 맞춰주세요.")
                else:
                    st.success("✅ 비중 확인 완료!")
                    
                    category_map = df_analysis.set_index('ETF명')['카테고리'].to_dict()
                    category_map['현금보유 (Cash)'] = '현금'
                    
                    target_weights = optimal_portfolio_full.set_index('ETF명')['추천비중(%)'].to_dict()
                    current_weights = edited_holdings.set_index('ETF명')['현재 비중(%)'].to_dict()
                    all_keys = set(target_weights.keys()).union(set(current_weights.keys()))
                    
                    action_plan = []
                    for etf in all_keys:
                        curr_w = current_weights.get(etf, 0.0)
                        tgt_w = target_weights.get(etf, 0.0)
                        diff = tgt_w - curr_w
                        
                        if abs(diff) < 0.1: action = "유지 (Hold) ⏸️"
                        elif diff > 0: action = "매수 (Buy) 🟢"
                        else: action = "매도 (Sell) 🔴"
                            
                        action_plan.append({
                            "카테고리": category_map.get(etf, "기타"), "ETF 종목명": etf,
                            "현재 내 비중": curr_w, "목표 비중": tgt_w, "조정 필요 비중": diff, "행동 지침": action
                        })
                        
                    df_actions = pd.DataFrame(action_plan).sort_values(by="조정 필요 비중")
                    st.dataframe(df_actions.style.format({
                        "현재 내 비중": "{:.1f}%", "목표 비중": "{:.1f}%", "조정 필요 비중": "{:+.1f}%"
                    }).map(lambda x: "color: red;" if "Sell" in x else ("color: green;" if "Buy" in x else "color: gray;"), subset=["행동 지침"]), use_container_width=True, hide_index=True)
else:
    st.sidebar.error(f"❌ `{csv_filename}` 파일이 경로에 필요합니다.")