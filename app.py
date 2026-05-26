import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# 웹페이지 기본 설정
st.set_page_config(layout="wide", page_title="GAPS ETF 투자 대회 대시보드")

st.title("🥇 GAPS ETF 대시보드 [V14 - 대회 규정 최적화 엔진]")
st.markdown("💡 위험자산(70%) 및 섹터별 투자 한도를 엄격히 준수하여 계좌 수익률을 극대화하는 **실전 비중 산출기**입니다.")

# --- [핵심] 대회 룰 기반 포트폴리오 최적화 알고리즘 ---
def optimize_portfolio(df_predictions, target_col='Pred'):
    # 기대수익률이 높은 순으로 정렬
    df_sorted = df_predictions.sort_values(by=target_col, ascending=False)
    
    portfolio = []
    total_budget = 100.0
    risk_budget = 70.0
    cat_alloc = {}
    
    for _, row in df_sorted.iterrows():
        if total_budget <= 0: break
        
        # 기대수익률이 마이너스면 차라리 현금을 보유하는 것이 유리하므로 매수 중단
        if row[target_col] <= 0: break 
        
        raw_cat = str(row['카테고리']).replace(' ', '').replace('_', '')
        
        # 카테고리별 대회 룰 매칭 (기본값 설정)
        limit, asset_type, c_name = 10, '위험', '기타주식'
        
        # 규정표 기준 한도 맵핑
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
        
        # 현재 해당 카테고리에 얼마나 할당했는지 체크
        current_cat_alloc = cat_alloc.get(c_name, 0.0)
        available_cat = limit - current_cat_alloc
        
        # 투입 가능 비중 계산
        weight = min(total_budget, available_cat)
        if asset_type == '위험':
            weight = min(weight, risk_budget)
            
        # 비중이 0 이상 할당 가능하다면 포트폴리오에 편입
        if weight > 0:
            item = {
                'ETF명': row['ETF명'], '카테고리': c_name, '자산군': asset_type,
                '추천비중(%)': weight, '기대수익률(%)': row[target_col]
            }
            if 'Actual' in row: item['실제수익률(%)'] = row['Actual']
            portfolio.append(item)
            
            total_budget -= weight
            if asset_type == '위험': risk_budget -= weight
            cat_alloc[c_name] = current_cat_alloc + weight

    # 예산이 남았다면 현금(안전자산)으로 보유
    if total_budget > 0:
        portfolio.append({
            'ETF명': '현금보유 (Cash)', '카테고리': '현금', '자산군': '안전',
            '추천비중(%)': total_budget, '기대수익률(%)': 0.0, '실제수익률(%)': 0.0 if 'Actual' in row else np.nan
        })
        
    return pd.DataFrame(portfolio)

@st.cache_data(ttl=21600, show_spinner="⏳ 최적화 알고리즘으로 백테스팅 및 비중 산출 중... (약 1분 소요)")
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
            df['Weight_Score'] = (df['Price_Change'] * 0.4 + df['Price_Change'].shift(1) * 0.3 + 
                                  df['Price_Change'].shift(2) * 0.2 + df['Price_Change'].shift(3) * 0.1)
            
            df['MA20'] = df['Price'].rolling(20).mean()
            df['Trend'] = np.where(df['Price'] >= df['MA20'], "🟢상승세", "🔴하락세")
            df['Next_Return'] = df['Price_Change'].shift(-1)
            df_clean = df.dropna(subset=['Weight_Score']).copy()

            df_for_pred = df_clean.dropna(subset=['Next_Return'])
            if len(df_for_pred) < 20: continue
            
            slope, intercept = np.polyfit(df_for_pred['Weight_Score'], df_for_pred['Next_Return'], 1)
            
            df_clean['Pred'] = (intercept + slope * df_clean['Weight_Score']) * 100
            df_clean['Actual'] = df_clean['Next_Return'] * 100
            
            df_hist = df_clean.dropna(subset=['Actual']).tail(20)
            for date, row in df_hist.iterrows():
                daily_records.append({
                    'Date': date.strftime('%Y-%m-%d'), 'ETF명': info['name'], '카테고리': info['category'],
                    'Pred': row['Pred'], 'Actual': row['Actual']
                })

            curr_score = df_clean['Weight_Score'].iloc[-1]
            pred_ret = (intercept + slope * curr_score) * 100
            corr = df_for_pred['Weight_Score'].corr(df_for_pred['Next_Return'])

            summary_results.append({
                '종목코드': 'A' + ticker, 'ETF명': info['name'], '카테고리': info['category'],
                '현재추세': df_clean['Trend'].iloc[-1], '마감 가중치 스코어': curr_score * 100,
                '오늘 종가 기대수익률': pred_ret, '상관성(모델신뢰도)': corr
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
        st.sidebar.success("📂 최적화 엔진 로드 완료!")
        
        tab1, tab2, tab3 = st.tabs(["🎯 오늘의 실전 매매 비중 (포트폴리오)", "🔥 대회 룰 적용 백테스트 (누적수익률)", "🔍 개별 종목 분석"])

        # [탭 1] 실전 매매 오더(Order)
        with tab1:
            st.subheader("🎯 오늘 자 최적화 포트폴리오 매수 비중 (대회 룰 100% 준수)")
            st.markdown("규정에 따라 **위험자산 최대 70%**, **각 그룹별 최대 비중**을 꽉 채워 기대수익률을 극대화한 오더입니다. 장 초반에 이 비율대로 세팅하세요.")
            
            # 오늘자 예측 데이터를 바탕으로 포트폴리오 최적화기 가동
            df_pred_today = df_analysis.rename(columns={'오늘 종가 기대수익률': 'Pred'})
            optimal_portfolio = optimize_portfolio(df_pred_today, target_col='Pred')
            
            # 요약 정보 (위험/안전 자산 비중 체크)
            risk_sum = optimal_portfolio[optimal_portfolio['자산군'] == '위험']['추천비중(%)'].sum()
            safe_sum = optimal_portfolio[optimal_portfolio['자산군'] == '안전']['추천비중(%)'].sum()
            expected_total_return = sum(optimal_portfolio['추천비중(%)']/100 * optimal_