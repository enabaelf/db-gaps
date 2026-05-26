import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# 웹페이지 기본 설정
st.set_page_config(layout="wide", page_title="GAPS ETF 투자 대회 대시보드")

st.title("🥇 GAPS ETF 대시보드 [V21 - 10개년 패턴 기댓값(손익비) 모델]")
st.markdown("💡 1~4일 전의 [상/하] 방향성 패턴을 분석하여, 단순 승률이 아닌 **과거 해당 패턴이 보였던 실제 기대수익률(확률 × 변동폭)**을 합산한 모델입니다.")

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
        if row[target_col] <= 0: break # 기대수익률이 0 이하(손실 예상)이면 패스
        
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
                '추천비중(%)': weight, '기대수익률(%)': row[target_col]
            }
            if has_actual: item['실제수익률(%)'] = row['Actual']
            portfolio.append(item)
            
            total_budget -= weight
            if asset_type == '위험': risk_budget -= weight
            cat_alloc[c_name] = current_cat_alloc + weight

    if total_budget > 0:
        portfolio.append({
            'ETF명': '현금보유 (Cash)', '카테고리': '현금', '자산군': '안전',
            '추천비중(%)': total_budget, '기대수익률(%)': 0.0, '실제수익률(%)': 0.0 if has_actual else np.nan
        })
        
    return pd.DataFrame(portfolio)

@st.cache_data(ttl=21600, show_spinner="⏳ 10년 치 상/하 패턴 전수조사 및 기댓값(손익비) 계산 중... (약 1분 소요)")
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
            if len(df) < 40: continue

            df['Price_Change'] = df['Price'].pct_change()
            
            # 1. 가격 방향 기호화 (상승: 1, 하락: 0) - 패턴 족보 생성용
            df['Dir'] = np.where(df['Price_Change'] > 0, 1, 0)
            df['L1'] = df['Dir']
            df['L2'] = df['Dir'].shift(1)
            df['L3'] = df['Dir'].shift(2)
            df['L4'] = df['Dir'].shift(3)
            
            # 예측 대상: 다음 날의 '실제 수익률' (확률이 아닌 기댓값 도출용)
            df['Next_Return'] = df['Price_Change'].shift(-1)
            
            df_clean = df.dropna(subset=['L4', 'Next_Return']).copy()
            if len(df_clean) < 100: continue
            
            # 3. 각 패턴별 '기대수익률(평균 수익률)' 족보 사전 구축
            e1_map = df_clean.groupby('L1')['Next_Return'].mean().to_dict()
            e2_map = df_clean.groupby(['L1', 'L2'])['Next_Return'].mean().to_dict()
            e3_map = df_clean.groupby(['L1', 'L2', 'L3'])['Next_Return'].mean().to_dict()
            e4_map = df_clean.groupby(['L1', 'L2', 'L3', 'L4'])['Next_Return'].mean().to_dict()
            
            # 4. 과거 매일의 데이터에 기댓값 대입 (희귀 패턴으로 결측 시 0.0 부여)
            df_clean['E1'] = df_clean['L1'].map(e1_map).fillna(0.0)
            df_clean['E2'] = df_clean.set_index(['L1', 'L2']).index.map(e2_map.get).fillna(0.0)
            df_clean['E3'] = df_clean.set_index(['L1', 'L2', 'L3']).index.map(e3_map.get).fillna(0.0)
            df_clean['E4'] = df_clean.set_index(['L1', 'L2', 'L3', 'L4']).index.map(e4_map.get).fillna(0.0)
            
            # 5. 네 가지 기댓값을 합산 후 정규화 (최종 평균 기대수익률 산출)
            df_clean['Avg_Exp_Return'] = (df_clean['E1'] + df_clean['E2'] + df_clean['E3'] + df_clean['E4']) / 4.0
            
            # 퍼센티지(%) 변환 및 포트폴리오 연동
            df_clean['Pred'] = df_clean['Avg_Exp_Return'] * 100
            df_clean['Actual'] = df_clean['Next_Return'] * 100
            
            df_hist = df_clean.dropna(subset=['Actual']).tail(20)
            for date, row in df_hist.iterrows():
                daily_records.append({
                    'Date': date.strftime('%Y-%m-%d'), 'ETF명': info['name'], '카테고리': info['category'],
                    'Pred': row['Pred'], 'Actual': row['Actual']
                })

            # 오늘 자 마감 기준 내일의 예측치
            final_exp_return = df_clean['Pred'].iloc[-1]
            df['MA20'] = df['Price'].rolling(20).mean()
            df['Trend'] = np.where(df['Price'] >= df['MA20'], "🟢상승세", "🔴하락세")

            summary_results.append({
                '종목코드': 'A' + ticker, 'ETF명': info['name'], '카테고리': info['category'],
                '현재추세': df['Trend'].iloc[-1], '패턴 기반 기대수익률': final_exp_return
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
        st.sidebar.success("📂 V21 패턴 기댓값 맵핑 완료")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 오늘의 실전 매매 비중", 
            "🏆 섹터별 추천 TOP 3", 
            "🔥 백테스트 (이론적 최대치 비교)", 
            "🔍 개별 종목 분석"
        ])

        # [탭 1] 실전 매매 오더
        with tab1:
            st.subheader("🎯 오늘 자 최적화 포트폴리오 비중")
            st.markdown("단순 승률이 아닌, 확률과 변동폭을 곱한 **기대수익률(기댓값)이 가장 높은 종목**을 우선순위로 채웠습니다.")
            
            df_pred_today = df_analysis.rename(columns={'패턴 기반 기대수익률': 'Pred'})
            optimal_portfolio = optimize_portfolio(df_pred_today, target_col='Pred')
            
            risk_sum = float(optimal_portfolio[optimal_portfolio['자산군'] == '위험']['추천비중(%)'].sum())
            safe_sum = float(optimal_portfolio[optimal_portfolio['자산군'] == '안전']['추천비중(%)'].sum())
            
            m1, m2 = st.columns(2)
            m1.metric("🔴 위험자산 총합 (Max 70%)", f"{risk_sum:.1f}%")
            m2.metric("🟢 안전자산 총합", f"{safe_sum:.1f}%")
            
            st.dataframe(optimal_portfolio[['자산군', '카테고리', 'ETF명', '추천비중(%)', '기대수익률(%)']].style.format({'추천비중(%)': '{:.1f}%', '기대수익률(%)': '{:.3f}%'}), use_container_width=True)

        # [탭 2] 섹터별 TOP 3
        with tab2:
            st.subheader("🏆 카테고리별 패턴 기대수익률 TOP 3")
            unique_cats = df_analysis['카테고리'].unique()
            for i in range(0, len(unique_cats), 2):
                cols = st.columns(2)
                for j, cat in enumerate(unique_cats[i:i+2]):
                    with cols[j]:
                        st.markdown(f"#### 📂 {cat}")
                        df_cat = df_analysis[df_analysis['카테고리'] == cat].sort_values(by='패턴 기반 기대수익률', ascending=False).head(3).reset_index(drop=True)
                        df_cat.index += 1
                        st.dataframe(df_cat[['ETF명', '현재추세', '패턴 기반 기대수익률']].style.format({'패턴 기반 기대수익률': '{:.3f}%'}), use_container_width=True)

        # [탭 3] 백테스트
        with tab3:
            st.subheader("🔥 규정 비율 매일 리밸런싱 백테스트 (기댓값 모델 vs 1/N vs 이론적 최대치)")
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
                c1.metric(f"🤖 {period} 패턴 기댓값 모델", f"{cum_model[-1]:.2f}%")
                c2.metric(f"📊 {period} 시장 평균 1/N", f"{cum_market[-1]:.2f}%")
                c3.metric("✨ 알파 (초과 수익)", f"{(cum_model[-1] - cum_market[-1]):.2f}%")
                c4.metric("👑 신의 영역 (이론적 최대)", f"{cum_max[-1]:.2f}%")
                
                df_chart = pd.DataFrame({
                    '👑 신의 영역 (완벽한 예지력)': cum_max,
                    '🤖 기댓값 최적화 포트폴리오': cum_model, 
                    '📊 시장 전체 평균 1/N': cum_market
                }, index=dates_list)
                st.line_chart(df_chart, use_container_width=True)

        # [탭 4] 종목 상세조회
        with tab4:
            st.subheader("🔍 ETF 개별 종목 정밀 분석 및 기술적 지표 차트")
            target_etf = st.selectbox("종목을 선택하세요:", df_analysis['ETF명'].unique())
            row = df_analysis[df_analysis['ETF명'] == target_etf].iloc[0]
            
            col_a, col_b = st.columns(2)
            col_a.metric("현재 추세", row['현재추세'])
            col_b.metric("내일 패턴 기반 기대수익률", f"{row['패턴 기반 기대수익률']:.3f}%")
            
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