import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# 웹페이지 기본 설정
st.set_page_config(layout="wide", page_title="GAPS ETF 투자 대회 대시보드")

st.title("📊 GAPS ETF 대시보드 [V10 - 마스터 버전]")
st.markdown("거래량 가중 모델 기반 수익률 최적화 및 기간별 성과 검증 시스템")

@st.cache_data(ttl=21600, show_spinner="⏳ 전 종목 10년치 주가 분석 및 1달치 시뮬레이션 수행 중... (약 1분 소요)")
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
    
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365*10)).strftime('%Y-%m-%d')

    for ticker, info in ticker_dict.items():
        try:
            df = fdr.DataReader(ticker, start_date, end_date)[['Close', 'Volume']].rename(columns={'Close': 'Price'})
            if len(df) < 40: continue

            df['Price_Change'] = df['Price'].pct_change()
            df['Vol_MA20'] = df['Volume'].rolling(20).mean()
            df['Vol_Ratio'] = np.clip(df['Volume'] / df['Vol_MA20'], 0.5, 3.0).fillna(1.0)
            df['VW_Change'] = df['Price_Change'] * df['Vol_Ratio']
            
            df['Weight_Score'] = (df['VW_Change'] * 0.4 + df['VW_Change'].shift(1) * 0.3 + 
                                  df['VW_Change'].shift(2) * 0.2 + df['VW_Change'].shift(3) * 0.1)
            
            df['MA20'] = df['Price'].rolling(20).mean()
            df['Trend'] = np.where(df['Price'] >= df['MA20'], "🟢상승세", "🔴하락세")
            df['Next_Return'] = df['Price_Change'].shift(-1)
            df_clean = df.dropna(subset=['Weight_Score'])

            # 회귀 분석 (전체 데이터)
            df_for_pred = df_clean.dropna(subset=['Next_Return'])
            slope, intercept = np.polyfit(df_for_pred['Weight_Score'], df_for_pred['Next_Return'], 1)
            curr_score = df_clean['Weight_Score'].iloc[-1]
            pred_ret = (intercept + slope * curr_score) * 100
            corr = df_for_pred['Weight_Score'].corr(df_for_pred['Next_Return'])

            # 기간별 실제 누적 수익률 (1주=5일, 2주=10일, 1달=20일)
            ret_1w = (df['Price'].iloc[-1] / df['Price'].iloc[-6] - 1) * 100 if len(df) >= 6 else 0
            ret_2w = (df['Price'].iloc[-1] / df['Price'].iloc[-11] - 1) * 100 if len(df) >= 11 else 0
            ret_1m = (df['Price'].iloc[-1] / df['Price'].iloc[-21] - 1) * 100 if len(df) >= 21 else 0

            # 시점별 모델 예측치 (과거 회귀선 기준)
            def get_old_pred(days_ago):
                df_past = df_clean.iloc[:-days_ago]
                if len(df_past) < 20: return np.nan
                s, i = np.polyfit(df_past.dropna(subset=['Next_Return'])['Weight_Score'], 
                                  df_past.dropna(subset=['Next_Return'])['Next_Return'], 1)
                return (i + s * df_clean['Weight_Score'].iloc[-days_ago]) * 100

            summary_results.append({
                '종목코드': 'A' + ticker, 'ETF명': info['name'], '카테고리': info['category'],
                '현재추세': df_clean['Trend'].iloc[-1], '마감 가중치 스코어': curr_score * 100,
                '내일 기대수익률': pred_ret, '상관성(모델신뢰도)': corr,
                'ret_5d': ret_1w, 'ret_10d': ret_2w, 'ret_20d': ret_1m,
                'pred_5d': get_old_pred(5), 'pred_10d': get_old_pred(10), 'pred_20d': get_old_pred(20)
            })
        except: pass
    return pd.DataFrame(summary_results)

csv_filename = "gaps_etf_list.csv"

if os.path.exists(csv_filename):
    df_raw = None
    for enc in ['utf-8-sig', 'cp949', 'utf-8', 'euc-kr']:
        try:
            df_raw = pd.read_csv(csv_filename, header=None, encoding=enc, dtype=str)
            break
        except: continue

    if df_raw is not None:
        df_analysis = run_full_analysis(df_raw)
        st.sidebar.success("📂 분석 완료")
        
        # 탭 구성
        tab1, tab2, tab3, tab4 = st.tabs(["🏆 섹션별 TOP 3", "🌲 모의투자 성과비교", "🔍 종목 상세조회", "📖 용어 가이드"])

        # [탭 1] 섹션별 추천
        with tab1:
            st.subheader("📈 내일 자 기대수익률 최적화 추천")
            unique_cats = df_analysis['카테고리'].unique()
            for i in range(0, len(unique_cats), 2):
                cols = st.columns(2)
                for j, cat in enumerate(unique_cats[i:i+2]):
                    with cols[j]:
                        st.markdown(f"#### 📂 {cat}")
                        df_cat = df_analysis[df_analysis['카테고리'] == cat].sort_values(by='내일 기대수익률', ascending=False).head(3).reset_index(drop=True)
                        df_cat.index += 1
                        st.dataframe(df_cat[['ETF명', '현재추세', '내일 기대수익률']].style.format({'내일 기대수익률': '{:.2f}%'}), use_container_width=True)

        # [탭 2] 성과 비교 (1주, 2주, 1달 선택)
        with tab2:
            st.subheader("🌲 과거 시점 모델 예측 vs 실제 성과 비교")
            period = st.radio("검증 기간 선택:", ["1주 (5영업일)", "2주 (10영업일)", "1달 (20영업일)"], horizontal=True)
            
            p_code = "5d" if "1주" in period else ("10d" if "2주" in period else "20d")
            
            df_bt = df_analysis.dropna(subset=[f'pred_{p_code}', f'ret_{p_code}'])
            df_model_picks = df_bt.sort_values(by=f'pred_{p_code}', ascending=False).head(5)
            df_market_winners = df_bt.sort_values(by=f'ret_{p_code}', ascending=False).head(5)
            
            m1, m2, m3 = st.columns(3)
            m1.metric(f"🤖 {period} 전 모델 추천 평균", f"{df_model_picks[f'ret_{p_code}'].mean():.2f}%")
            m2.metric(f"👑 {period} 실제 시장 TOP 5 평균", f"{df_market_winners[f'ret_{p_code}'].mean():.2f}%")
            m3.metric(f"📊 {period} 전체 ETF 평균", f"{df_bt[f'ret_{p_code}'].mean():.2f}%")
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"#### 🤖 {period} 전 모델의 원픽들")
                st.dataframe(df_model_picks[['ETF명', f'pred_{p_code}', f'ret_{p_code}']].rename(columns={f'pred_{p_code}': '당시 예측치', f'ret_{p_code}': '실제 누적수익률'}).style.format({'당시 예측치': '{:.2f}%', '실제 누적수익률': '{:.2f}%'}), use_container_width=True)
            with c2:
                st.markdown(f"#### 👑 {period} 동안 실제 수익률 1위들")
                st.dataframe(df_market_winners[['ETF명', f'ret_{p_code}', f'pred_{p_code}']].rename(columns={f'pred_{p_code}': '당시 예측치', f'ret_{p_code}': '실제 누적수익률'}).style.format({'당시 예측치': '{:.2f}%', '실제 누적수익률': '{:.2f}%'}), use_container_width=True)

        # [탭 3] 종목 상세조회 (복구 완료!)
        with tab3:
            st.subheader("🔍 ETF 개별 종목 정밀 분석")
            target_etf = st.selectbox("종목을 선택하세요:", df_analysis['ETF명'].unique())
            row = df_analysis[df_analysis['ETF명'] == target_etf].iloc[0]
            
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("현재 추세", row['현재추세'])
            col_b.metric("내일 예측 수익률", f"{row['내일 기대수익률']:.2f}%")
            col_c.metric("상관성 (모델 신뢰도)", f"{row['상관성(모델신뢰도)']:.2f}")
            
            st.markdown("---")
            st.markdown("#### 📈 최근 1년 주가 흐름")
            raw_t = row['종목코드'].replace('A', '')
            try:
                df_chart = fdr.DataReader(raw_t, (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d'))[['Close']].rename(columns={'Close': '주가'})
                st.line_chart(df_chart, use_container_width=True)
            except: st.error("차트를 불러올 수 없습니다.")

        # [탭 4] 용어 가이드 (팀원 설명용)
        with tab4:
            st.subheader("📖 대시보드 용어 및 원리 가이드")
            with st.expander("1. 마감 가중치 스코어 (Closing Weight Score)란?"):
                st.write("단순히 올랐다 내렸다가 아니라, **거래량이 터지면서 강하게 움직인 날에 가중치**를 더 준 점수입니다. 최근 4일간의 에너지를 40%, 30%, 20%, 10% 비율로 합산하여 계산합니다.")
            with st.expander("2. 내일 기대 예측수익률 (Expected Return)이란?"):
                st.write("과거 10년 동안 현재와 비슷한 '가중치 스코어'가 발생했을 때, **그다음 날 주가가 실제로 평균 몇 %나 움직였는지** 통계적으로 추정한 값입니다.")
            with st.expander("3. 상관성 (Correlation)이란?"):
                st.write("이 종목이 우리 모델의 예측을 얼마나 정직하게 잘 따르는지를 나타내는 수치입니다. **1에 가까울수록 모델이 예측한 대로 움직일 가능성이 높다는 신뢰의 지표**입니다.")
            with st.expander("4. 성과 비교 탭은 어떻게 보나요?"):
                st.write("과거 1주/2주/1달 전으로 돌아가서 '그 당시 모델의 눈'으로 종목을 뽑아보고, 그 종목들이 실제로 오늘까지 얼마나 수익을 냈는지 '진짜 정답'과 비교하는 공간입니다.")