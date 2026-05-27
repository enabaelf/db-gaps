import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# 웹페이지 기본 설정
st.set_page_config(layout="wide", page_title="GAPS ETF 투자 대회 대시보드")

st.title("🥇 GAPS ETF 대시보드 [V23 - 소르티노-레짐 하방 방어 모델]")
st.markdown("💡 1~4일 패턴 기댓값 앙상블에 **[20일 추세 필터]**와 **[하방 변동성 페널티]**를 융합하여, 하락장 낙폭(MDD)을 극도로 제어하는 방어형 모델입니다.")

# --- 대회 룰 기반 포트폴리오 최적화 알고리즘 ---
def optimize_portfolio(df_predictions, target_col='Pred'):
    df_sorted = df_predictions.sort_values(by=target_col, ascending=False)
    
    portfolio = []
    total_budget = 100.0
    risk_budget = 70.0
    cat_alloc = {}
    
    has_actual = 'Actual' in df_predictions.columns
    
    for _, row in df_sorted.iterrows():
        if total_budget <= 0: break
        if row[target_col] <= 0: break # 추세 탈락 및 위험조정점수 0 이하 종목은 매수 안 함 -> 현금화 유도
        
        raw_cat = str(row['카테고리']).replace(' ', '').replace('_', '')
        limit, asset_type, c_name = 10, '위험', '기타주식'
        
        if '국내' in raw_cat and '주식' in raw_cat and '지수' in raw_cat: limit, asset_type, c_name = 30, '위험', '국내주식_지수'
        elif '국내' in raw_cat and '주식' in raw_cat and '섹터' in raw_cat: limit, asset_type, c_name = 15, '위험', '국내주식_섹터'
        elif '해외' in raw_cat and '주식' in raw_cat and '지수' in raw_cat: limit, asset_type, c_name = 30, '위험', '해외주식_지수'
        elif '해외' in raw_cat and '주식' in raw_cat and '섹터' in raw_cat: limit, asset_type, c_name = 10, '위험', '해외주식_섹터'
        elif 'FX' in raw_cat.upper() or '원자재' in raw_cat: limit, asset_type, c_name = 20, '위험', 'FX_및_원자재'
        elif '국내' in raw_cat and '채권' in raw_cat and '종합' in raw_cat: limit, asset_type, c_name = 50, '안전', '국내채권_종합'
        elif '국내' in raw_cat and '채권' in raw_cat and '회사' in raw_cat: limit, asset_type, c_name = 30, '안전', '국내채권_회사채'
        elif '해외' in raw_cat and '채권' in raw_cat and '종합' in raw_cat: limit, asset_type, c_name = 50, '안전', '해외채권_종합'
        elif '해외' in raw_cat and '채권' in raw_cat and '회사' in raw_cat: limit, asset_type, c_name = 30, '안전', '해외채권_회사채'
        elif '단기' in raw_cat or '금리' in raw_cat: limit, asset_type, c_name = 50, '안전', '단기채권'
        elif '채권' in raw_cat: limit, asset_type, c_name = 50, '안전', '기타_안전채권'
        elif '주식' in raw_cat: limit, asset_type, c_name = 10, '위험', '기타_위험주식'
        
        current_cat_alloc = cat_alloc.get(c_name, 0.0)
        available_cat = limit - current_cat_alloc
        
        weight = min(total_budget, available_cat)
        if asset_type == '위험':
            weight = min(weight, risk_budget)
            
        if weight > 0:
            item = {
                'ETF명': row['ETF명'], '카테고리': c_name, '자산군': asset_type,
                '추천비중(%)': weight, '위험조정점수': row[target_col]
            }
            if has_actual: item['실제수익률(%)'] = row['Actual']
            portfolio.append(item)
            
            total_budget -= weight
            if asset_type == '위험': risk_budget -= weight
            cat_alloc[c_name] = current_cat_alloc + weight

    if total_budget > 0:
        portfolio.append({
            'ETF명': '현금보유 (Cash)', '카테고리': '현금', '자산군': '안전',
            '추천비중(%)': total_budget, '위험조정점수': 0.0, '실제수익률(%)': 0.0 if has_actual else np.nan
        })
        
    return pd.DataFrame(portfolio)

@st.cache_data(ttl=21600, show_spinner="⏳ 앙상블 연산 및 하방 리스크 필터링 적용 중... (약 1분 소요)")
def run_full_analysis(df_raw):
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
        n_col = next((str(c) for c in df_data.columns if 'ETF' in str(c) or '명' in str(c)), str(df_data.columns[1]) if len(df_data.columns) > 1 else str(df_data.columns[0]))
        c2_col = next((str(c) for c in df_data.columns if '구분2' in str(c)), str(df_data.columns[-1]))

        for _, row in df_data.iterrows():
            ticker = str(row[t_col]).strip()
            if ticker.upper().startswith('A'): ticker = ticker[1:]
            if len(ticker) == 6 and ticker.isalnum():
                ticker_dict[ticker] = {'name': str(row[n_col]).strip(), 'category': str(row[c2_col]).strip()}

    summary_results = []
    daily_records = []
    
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365*10)).strftime('%Y-%m-%d')

    for ticker, info in ticker_dict.items():
        try:
            df = fdr.DataReader(ticker, start_date, end_date)[['Close']].rename(columns={'Close': 'Price'})
            if len(df) < 65: continue

            # --- [기술적 지표 사전 계산] ---
            df['Price_Change'] = df['Price'].pct_change()
            df['MA20'] = df['Price'].rolling(20).mean()
            
            # 옵션 2: 하방 변동성 (음수 수익률만 추출하여 최근 60일간 표준편차 계산)
            df['Downside_Return'] = df['Price_Change'].clip(upper=0)
            df['Downside_Vol'] = df['Downside_Return'].rolling(60).std()

            df['Dir'] = np.where(df['Price_Change'] > 0, 1, 0)
            df['L1'] = df['Dir']
            df['L2'] = df['Dir'].shift(1)
            df['L3'] = df['Dir'].shift(2)
            df['L4'] = df['Dir'].shift(3)
            df['Next_Return'] = df['Price_Change'].shift(-1)
            
            # 지표들이 모두 채워진 시점부터 클리닝
            df_clean = df.dropna(subset=['L4', 'Next_Return', 'MA20', 'Downside_Vol']).copy()
            if len(df_clean) < 100: continue
            
            # 1. 패턴 기댓값 맵핑
            e1_map = df_clean.groupby('L1')['Next_Return'].mean().to_dict()
            e2_map = df_clean.groupby(['L1', 'L2'])['Next_Return'].mean().to_dict()
            e3_map = df_clean.groupby(['L1', 'L2', 'L3'])['Next_Return'].mean().to_dict()
            e4_map = df_clean.groupby(['L1', 'L2', 'L3', 'L4'])['Next_Return'].mean().to_dict()
            
            df_clean['E1'] = df_clean['L1'].map(e1_map).fillna(0.0)
            df_clean['E2'] = df_clean.set_index(['L1', 'L2']).index.map(e2_map.get).fillna(0.0)
            df_clean['E3'] = df_clean.set_index(['L1', 'L2', 'L3']).index.map(e3_map.get).fillna(0.0)
            df_clean['E4'] = df_clean.set_index(['L1', 'L2', 'L3', 'L4']).index.map(e4_map.get).fillna(0.0)
            
            # 2. 최근 500일 회귀 가중치 도출
            fit_window = min(len(df_clean), 500)
            df_fit = df_clean.tail(fit_window)
            X_fit = df_fit[['E1', 'E2', 'E3', 'E4']].values
            y_fit = df_fit['Next_Return'].values
            
            X_design = np.column_stack((np.ones(len(X_fit)), X_fit))
            coeffs, _, _, _ = np.linalg.lstsq(X_design, y_fit, rcond=None)
            intercept = coeffs[0]
            w1, w2, w3, w4 = coeffs[1:]
            
            # 3. 앙상블 기댓값 산출 (%)
            df_clean['Final_Pred'] = (intercept + df_clean['E1']*w1 + df_clean['E2']*w2 + df_clean['E3']*w3 + df_clean['E4']*w4) * 100
            
            # --- [핵심 고도화: 옵션 2 + 3 융합 하방 방어 필터링] ---
            df_clean['Is_Uptrend'] = np.where(df_clean['Price'] >= df_clean['MA20'], 1, 0)
            df_clean['Downside_Vol'] = df_clean['Downside_Vol'].replace(0, 1e-5).fillna(1e-5) # 0 나누기 방지
            
            # 조건식: 20일선 위에 있고(Is_Uptrend==1) 기댓값이 양수일 때만 [기댓값 / 하방변동성] 점수 부여, 하락세면 무조건 0점(현금행)
            df_clean['Risk_Adjusted_Score'] = np.where(
                (df_clean['Is_Uptrend'] == 1) & (df_clean['Final_Pred'] > 0),
                df_clean['Final_Pred'] / (df_clean['Downside_Vol'] * 100),
                0.0
            )
            
            df_clean['Pred'] = df_clean['Risk_Adjusted_Score']
            df_clean['Actual'] = df_clean['Next_Return'] * 100
            
            df_hist = df_clean.dropna(subset=['Actual']).tail(20)
            for date, row in df_hist.iterrows():
                daily_records.append({
                    'Date': date.strftime('%Y-%m-%d'), 'ETF명': info['name'], '카테고리': info['category'],
                    'Pred': row['Pred'], 'Actual': row['Actual']
                })

            final_score = df_clean['Pred'].iloc[-1]
            trend_str = "🟢상승세" if df_clean['Is_Uptrend'].iloc[-1] == 1 else "🔴하락세(진입금지)"

            summary_results.append({
                '종목코드': 'A' + ticker, 'ETF명': info['name'], '카테고리': info['category'],
                '현재추세': trend_str, '위험조정점수': final_score
            })
        except: pass
        
    return pd.DataFrame(summary_results), pd.DataFrame(daily_records)

csv_filename = "gaps_etf_list.csv"

if os.path.exists(csv_filename):
    df_raw = None
    for enc in ['utf-8-sig', 'cp949', 'utf-8', 'euc-kr']:
        try:
            df_raw = pd.read_csv(csv_filename, header=None, encoding=enc, dtype=str)
            break
        except: continue

    if df_raw is not None:
        df_analysis, df_daily = run_full_analysis(df_raw)
        st.sidebar.success("📂 V23 소르티노-레짐 엔진 장착 완료")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 오늘의 실전 매매 비중", 
            "🏆 섹터별 추천 TOP 3", 
            "🔥 백테스트 (하방 방어력 검증)", 
            "🔍 개별 종목 분석"
        ])

        with tab1:
            st.subheader("🎯 오늘 자 최적화 포트폴리오 비중")
            st.markdown("⚠️ **주의**: 하락추세에 진입한 종목은 원천 배제되어 **현금 비중**이 자동으로 늘어날 수 있습니다.")
            df_pred_today = df_analysis.rename(columns={'위험조정점수': 'Pred'})
            optimal_portfolio = optimize_portfolio(df_pred_today, target_col='Pred')
            
            risk_sum = float(optimal_portfolio[optimal_portfolio['자산군'] == '위험']['추천비중(%)'].sum())
            safe_sum = float(optimal_portfolio[optimal_portfolio['자산군'] == '안전']['추천비중(%)'].sum())
            
            m1, m2 = st.columns(2)
            m1.metric("🔴 위험자산 총합 (Max 70%)", f"{risk_sum:.1f}%")
            m2.metric("🟢 안전자산 총합", f"{safe_sum:.1f}%")
            
            st.dataframe(optimal_portfolio[['자산군', '카테고리', 'ETF명', '추천비중(%)', '위험조정점수']].style.format({'추천비중(%)': '{:.1f}%', '위험조정점수': '{:.2f}'}), use_container_width=True)

        with tab2:
            st.subheader("🏆 카테고리별 위험조정점수 TOP 3")
            unique_cats = df_analysis['카테고리'].unique()
            for i in range(0, len(unique_cats), 2):
                cols = st.columns(2)
                for j, cat in enumerate(unique_cats[i:i+2]):
                    with cols[j]:
                        st.markdown(f"#### 📂 {cat}")
                        df_cat = df_analysis[df_analysis['카테고리'] == cat].sort_values(by='위험조정점수', ascending=False).head(3).reset_index(drop=True)
                        df_cat.index += 1
                        st.dataframe(df_cat[['ETF명', '현재추세', '위험조정점수']].style.format({'위험조정점수': '{:.2f}'}), use_container_width=True)

        with tab3:
            st.subheader("🔥 규정 비율 매일 리밸런싱 백테스트 (방어력 집중 검증)")
            period = st.radio("시뮬레이션 기간 선택:", ["1일 (1영업일)", "1주 (5영업일)", "2주 (10영업일)", "1달 (20영업일)"], horizontal=True)
            days = 1 if "1일" in period else (5 if "1주" in period else (10 if "2주" in period else 20))
            
            if not df_daily.empty:
                unique_dates = sorted(df_daily['Date'].unique())[-days:]
                df_period = df_daily[df_daily['Date'].isin(unique_dates)]
                
                daily_model_returns, daily_market_returns, daily_max_returns, dates_list = [], [], [], []
                
                for d in unique_dates:
                    df_d = df_period[df_period['Date'] == d]
                    past_portfolio = optimize_portfolio(df_d, target_col='Pred')
                    daily_port_return = float((past_portfolio['추천비중(%)'] / 100 * past_portfolio['실제수익률(%)'].fillna(0)).sum())
                    daily_model_returns.append(daily_port_return / 100)
                    
                    daily_market_returns.append(float(df_d['Actual'].mean() / 100))
                    
                    max_portfolio = optimize_portfolio(df_d, target_col='Actual')
                    daily_max_return = float((max_portfolio['추천비중(%)'] / 100 * max_portfolio['실제수익률(%)'].fillna(0)).sum())
                    daily_max_returns.append(daily_max_return / 100)
                    dates_list.append(d)
                
                cum_model = (np.cumprod(1 + np.array(daily_model_returns)) - 1) * 100
                cum_market = (np.cumprod(1 + np.array(daily_market_returns)) - 1) * 100
                cum_max = (np.cumprod(1 + np.array(daily_max_returns)) - 1) * 100
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(f"🛡️ {period} 방어형 앙상블 모델", f"{cum_model[-1]:.2f}%")
                c2.metric(f"📊 {period} 시장 평균", f"{cum_market[-1]:.2f}%")
                c3.metric("✨ 알파 (초과 수익)", f"{(cum_model[-1] - cum_market[-1]):.2f}%")
                c4.metric("👑 신의 영역 (이론적 최대)", f"{cum_max[-1]:.2f}%")
                
                df_chart = pd.DataFrame({
                    '👑 신의 영역 (완벽한 예지력)': cum_max,
                    '🛡️ 하방방어 융합 포트폴리오': cum_model, 
                    '📊 시장 전체 평균 1/N': cum_market
                }, index=dates_list)
                st.line_chart(df_chart, use_container_width=True)

        with tab4:
            st.subheader("🔍 ETF 개별 종목 정밀 분석")
            target_etf = st.selectbox("종목을 선택하세요:", df_analysis['ETF명'].unique())
            row = df_analysis[df_analysis['ETF명'] == target_etf].iloc[0]
            
            col_a, col_b = st.columns(2)
            col_a.metric("현재 추세", row['현재추세'])
            col_b.metric("오늘의 위험조정점수", f"{row['위험조정점수']:.2f}")
            
            ticker_clean = str(row['종목코드']).replace('A', '')
            try:
                with st.spinner(f"📈 {target_etf} 차트 로드 중..."):
                    g_start = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
                    g_end = datetime.today().strftime('%Y-%m-%d')
                    df_stock_chart = fdr.DataReader(ticker_clean, g_start, g_end)[['Close']].rename(columns={'Close': '마감 종가'})
                    df_stock_chart['20일 이동평균선'] = df_stock_chart['마감 종가'].rolling(20).mean()
                    st.line_chart(df_stock_chart, use_container_width=True)
            except Exception as e:
                st.error(f"⚠️ 차트 로드 실패: {e}")
else:
    st.sidebar.error(f"❌ `{csv_filename}` 파일을 찾을 수 없습니다.")