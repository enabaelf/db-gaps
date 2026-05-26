import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# 웹페이지 기본 설정
st.set_page_config(layout="wide", page_title="GAPS ETF 투자 대회 대시보드")

st.title("📊 GAPS ETF 대시보드 [V3 - 자동 로드 완료]")
st.markdown("과거 10년 주가 기반 상승 확률 및 기댓값 최적화 원픽 추천 시스템")

def to_numeric(val):
    if pd.isna(val) or val == '-':
        return np.nan
    return float(str(val).replace('%', ''))

@st.cache_data(show_spinner="⏳ 바탕화면의 CSV 파일을 읽어 10년 치 주가 분석 중... (약 1분 소요)")
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
            name = str(row[n_col]).strip()
            category = str(row[c2_col]).strip()

            if ticker.upper().startswith('A'): ticker = ticker[1:]
            if len(ticker) == 6 and ticker.isalnum():
                ticker_dict[ticker] = {'name': name, 'category': category}

    summary_results = []
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365*10)).strftime('%Y-%m-%d')

    for ticker, info in ticker_dict.items():
        try:
            df = fdr.DataReader(ticker, start_date, end_date)[['Close']].rename(columns={'Close': 'Price'})
            if len(df) < 5: continue

            df['Price_Change'] = df['Price'].pct_change()
            df['Is_Up'] = (df['Price_Change'] > 0).astype(int)
            df['Up_1d_ago'] = df['Is_Up'].shift(1)
            df['Up_2d_ago'] = df['Is_Up'].shift(2)
            df['Up_3d_ago'] = df['Is_Up'].shift(3)
            df = df.dropna()

            prob_1d = df[df['Up_1d_ago'] == 1]['Is_Up'].mean() * 100
            c_3u = (df['Up_1d_ago'] == 1) & (df['Up_2d_ago'] == 1) & (df['Up_3d_ago'] == 1)
            prob_3u = df[c_3u]['Is_Up'].mean() * 100 if len(df[c_3u]) > 0 else np.nan
            c_3d = (df['Up_1d_ago'] == 0) & (df['Up_2d_ago'] == 0) & (df['Up_3d_ago'] == 0)
            prob_3d = df[c_3d]['Is_Up'].mean() * 100 if len(df[c_3d]) > 0 else np.nan

            ret_1d = df[df['Up_1d_ago'] == 1]['Price_Change'].mean() * 100
            ret_3u = df[c_3u]['Price_Change'].mean() * 100 if len(df[c_3u]) > 0 else np.nan
            ret_3d = df[c_3d]['Price_Change'].mean() * 100 if len(df[c_3d]) > 0 else np.nan

            summary_results.append({
                '종목코드': 'A' + ticker, 'ETF명': info['name'], '카테고리': info['category'],
                '어제상승_오늘상승확률': f"{prob_1d:.2f}%" if not np.isnan(prob_1d) else "-",
                '어제상승_오늘평균수익률': f"{ret_1d:.2f}%" if not np.isnan(ret_1d) else "-",
                '3일하락_반등확률': f"{prob_3d:.2f}%" if not np.isnan(prob_3d) else "-",
                '3일하락_평균반등폭': f"{ret_3d:.2f}%" if not np.isnan(ret_3d) else "-",
                '3일상승_추가상승확률': f"{prob_3u:.2f}%" if not np.isnan(prob_3u) else "-",
                '3일상승_평균추가상승폭': f"{ret_3u:.2f}%" if not np.isnan(ret_3u) else "-",
                '분석일수(샘플)': f"{len(df)}일"
            })
        except: pass

    df_final = pd.DataFrame(summary_results)
    for col in ['어제상승_오늘상승확률', '어제상승_오늘평균수익률', '3일하락_반등확률', '3일하락_평균반등폭', '3일상승_추가상승확률', '3일상승_평균추가상승폭']:
        df_final[col + '_숫자'] = df_final[col].apply(to_numeric)
    return df_final

# 고정할 파일 이름 설정
csv_filename = "gaps_etf_list.csv"

# 파일이 같은 폴더(바탕화면)에 있는지 확인
if os.path.exists(csv_filename):
    st.sidebar.success(f"📂 `{csv_filename}` 파일 감지 완료!")
    
    # 파일 자동 읽기
    df_raw = None
    for enc in ['utf-8-sig', 'cp949', 'utf-8', 'euc-kr']:
        try:
            df_raw = pd.read_csv(csv_filename, header=None, encoding=enc, dtype=str)
            break
        except: continue

    if df_raw is not None:
        df_analysis = run_full_analysis(df_raw)
        st.sidebar.info(f"📊 총 {len(df_analysis)}개 종목 분석 완료")
        
        tab1, tab2 = st.tabs(["🏆 전략별 카테고리 원픽", "🔍 종목 상세조회"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🧊 [역발상] 과매도 원픽 (3일하락 후 반등폭 1위)")
                idx_best_drop = df_analysis.groupby('카테고리')['3일하락_평균반등폭_숫자'].idxmax().dropna()
                st.dataframe(df_analysis.loc[idx_best_drop, ['카테고리', '종목코드', 'ETF명', '3일하락_반등확률', '3일하락_평균반등폭']], use_container_width=True)
            with col2:
                st.subheader("🔥 [모멘텀] 상승세 원픽 (3일상승 후 추가폭 1위)")
                idx_best_rise = df_analysis.groupby('카테고리')['3일상승_평균추가상승폭_숫자'].idxmax().dropna()
                st.dataframe(df_analysis.loc[idx_best_rise, ['카테고리', '종목코드', 'ETF명', '3일상승_추가상승확률', '3일상승_평균추가상승폭']], use_container_width=True)
            
            st.subheader("📋 전체 ETF 데이터 분석 리포트")
            st.dataframe(df_analysis[['카테고리', '종목코드', 'ETF명', '어제상승_오늘상승확률', '어제상승_오늘평균수익률', '3일하락_반등확률', '3일하락_평균반등폭', '3일상승_추가상승확률', '3일상승_평균추가상승폭', '분석일수(샘플)']], use_container_width=True)
else:
    st.sidebar.error(f"❌ `{csv_filename}` 파일을 찾을 수 없습니다.")
    st.error(f"⚠️ **안내:** 바탕화면에 **`gaps_etf_list.csv`** 파일이 없습니다. 파일 이름을 똑같이 맞춰서 `app.py`와 같은 바탕화면에 넣어두시면 주르륵 자동으로 실행됩니다!")