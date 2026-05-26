import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# 웹페이지 기본 설정
st.set_page_config(layout="wide", page_title="GAPS ETF 투자 대회 대시보드")

st.title("📊 GAPS ETF 대시보드 [V8 - 섹터별 승률 검증 완료]")
st.markdown("최근 4일 변동폭 가중치(WMA) 및 10년 통계 기반 기대수익률 & 백테스트 시스템")

@st.cache_data(ttl=21600, show_spinner="⏳ 선형 회귀 학습 및 20일 치 전체 백테스팅 수행 중... (약 1분 소요)")
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
    backtest_records = []
    
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365*10)).strftime('%Y-%m-%d')

    for ticker, info in ticker_dict.items():
        try:
            df = fdr.DataReader(ticker, start_date, end_date)[['Close']].rename(columns={'Close': 'Price'})
            if len(df) < 30: continue

            df['Price_Change'] = df['Price'].pct_change()
            df['Weight_Score'] = (
                df['Price_Change'] * 0.4 +
                df['Price_Change'].shift(1) * 0.3 +
                df['Price_Change'].shift(2) * 0.2 +
                df['Price_Change'].shift(3) * 0.1
            )
            df['Next_Return'] = df['Price_Change'].shift(-1)
            df_clean = df.dropna(subset=['Weight_Score'])
            
            if len(df_clean) < 20: continue

            # --- [1] 내일의 기대수익률을 위한 전체 데이터 회귀분석 ---
            # 최신 데이터를 제외하지 않고 전체를 학습하여 오늘 자 예측에 사용
            df_for_pred = df_clean.dropna(subset=['Next_Return'])
            slope_all, intercept_all = np.polyfit(df_for_pred['Weight_Score'], df_for_pred['Next_Return'], 1)
            
            current_score = df_clean['Weight_Score'].iloc[-1]
            predicted_next_return = (intercept_all + slope_all * current_score) * 100
            correlation = df_for_pred['Weight_Score'].corr(df_for_pred['Next_Return'])

            summary_results.append({
                '종목코드': 'A' + ticker, 
                'ETF명': info['name'], 
                '카테고리': info['category'],
                '현재 가중치 스코어': f"{current_score*100:.3f}%",
                '내일 기대수익률': round(predicted_next_return, 4),
                '모델 신뢰도(상관성)': round(correlation, 2),
                '분석일수(샘플)': f"{len(df_for_pred)}일"
            })
            
            # --- [2] 백테스팅을 위한 분리 검증 (최근 20영업일) ---
            # 최근 20일을 테스트셋으로, 그 이전 데이터만 학습셋(Train)으로 사용하여 순수 예측력 검증
            train_df = df_clean.iloc[:-20].dropna(subset=['Next_Return'])
            test_df = df_clean.iloc[-20:]
            
            if len(train_df) > 10:
                slope_bt, intercept_bt = np.polyfit(train_df['Weight_Score'], train_df['Next_Return'], 1)
                
                for date, row in test_df.iterrows():
                    actual_return = row['Next_Return']
                    if pd.isna(actual_return): continue # 내일 주가가 아직 없는 오늘 자 데이터 패스
                    
                    pred_return = intercept_bt + slope_bt * row['Weight_Score']
                    
                    # 방향 맞춤 여부 (둘 다 양수거나 둘 다 음수면 적중)
                    is_hit = 1 if (pred_return > 0 and actual_return > 0) or (pred_return < 0 and actual_return < 0) else 0
                    
                    backtest_records.append({
                        '날짜': date.strftime('%Y-%m-%d'),
                        '카테고리': info['category'],
                        'ETF명': info['name'],
                        '예측수익률': pred_return * 100,
                        '실제수익률': actual_return * 100,
                        '적중': is_hit
                    })
        except: pass

    df_final = pd.DataFrame(summary_results)
    df_backtest = pd.DataFrame(backtest_records)
    return df_final, df_backtest

csv_filename = "gaps_etf_list.csv"

if os.path.exists(csv_filename):
    df_raw = None
    for enc in ['utf-8-sig', 'cp949', 'utf-8', 'euc-kr']:
        try:
            df_raw = pd.read_csv(csv_filename, header=None, encoding=enc, dtype=str)
            break
        except: continue

    if df_raw is not None:
        df_analysis, df_backtest = run_full_analysis(df_raw)
        st.sidebar.success(f"📂 분석 완료! (종목: {len(df_analysis)}개)")
        
        tab1, tab2, tab3 = st.tabs(["🏆 섹션별 기대수익률 TOP 3", "🔍 종목 상세조회", "🔬 기간/섹터별 승률 검증"])

        # [탭 1] 메인 화면
        with tab1:
            st.markdown("### 📈 내일 자 기대수익률 최적화 포트폴리오 추천")
            unique_categories = df_analysis['카테고리'].unique()
            cat_chunks = [unique_categories[i:i + 2] for i in range(0, len(unique_categories), 2)]
            
            for chunk in cat_chunks:
                cols = st.columns(2)
                for i, cat in enumerate(chunk):
                    with cols[i]:
                        st.subheader(f"📂 {cat} Top 3")
                        df_cat = df_analysis[df_analysis['카테고리'] == cat]
                        df_cat_top3 = df_cat.sort_values(by='내일 기대수익률', ascending=False).head(3).reset_index(drop=True)
                        df_cat_top3.index = df_cat_top3.index + 1
                        
                        df_display = df_cat_top3[['ETF명', '현재 가중치 스코어', '내일 기대수익률']].copy()
                        df_display['내일 기대수익률'] = df_display['내일 기대수익률'].apply(lambda x: f"{x:.2f}%")
                        st.dataframe(df_display, use_container_width=True)

        # [탭 2] 상세 조회
        with tab2:
            st.subheader("🔍 ETF 개별 종목 정밀 조회")
            if not df_analysis.empty:
                selected_name = st.selectbox("조회할 ETF 종목 선택:", df_analysis['ETF명'].tolist(), key="tab2_select")
                etf_info = df_analysis[df_analysis['ETF명'] == selected_name].iloc[0]
                
                m1, m2, m3 = st.columns(3)
                m1.metric("현재 마감 가중치 스코어(WMA)", f"{etf_info['현재 가중치 스코어']}")
                m2.metric("★ 내일 예측 기대수익률", f"{etf_info['내일 기대수익률']:.2f}%")
                m3.metric("모델 신뢰도 (상관성)", f"{etf_info['모델 신뢰도(상관성)']}")

        # [탭 3] 전체/섹터별 백테스팅 분석 (회원님 요청 기능)
        with tab3:
            st.subheader("🔬 회귀 모델 아웃오브샘플 전체 승률 검증")
            st.markdown("전체 ETF 종목을 대상으로 지정한 기간 동안 **모델의 '상승/하락' 예측이 실제 종가와 얼마나 일치했는지 성공률(Hit Rate)**을 계산합니다.")
            
            if not df_backtest.empty:
                # 기간 선택 라디오 버튼
                period_choice = st.radio(
                    "검증 기간을 선택하세요 (영업일 기준):", 
                    ["1주 (최근 5영업일)", "2주 (최근 10영업일)", "1달 (최근 20영업일)"],
                    horizontal=True
                )
                
                # 선택한 기간에 맞춰 날짜 필터링
                days_to_keep = 5 if "1주" in period_choice else (10 if "2주" in period_choice else 20)
                unique_dates = sorted(df_backtest['날짜'].unique())[-days_to_keep:]
                df_filtered_bt = df_backtest[df_backtest['날짜'].isin(unique_dates)]
                
                # 전체 예측 성공률 계산
                total_cases = len(df_filtered_bt)
                total_hits = df_filtered_bt['적중'].sum()
                overall_hit_rate = (total_hits / total_cases) * 100 if total_cases > 0 else 0
                
                st.markdown(f"### 🎯 전체 유니버스 예측 성공률: **{overall_hit_rate:.1f}%**")
                st.caption(f"총 {total_cases}번의 예측 중 {total_hits}번 방향성 적중 (검증 기간: {unique_dates[0]} ~ {unique_dates[-1]})")
                
                st.markdown("---")
                
                # 섹터별 성공률 계산
                st.subheader("📊 섹터(카테고리)별 예측 성공률 순위")
                sector_stats = df_filtered_bt.groupby('카테고리').agg(
                    총예측횟수=('적중', 'count'),
                    적중횟수=('적중', 'sum')
                ).reset_index()
                sector_stats['승률(%)'] = (sector_stats['적중횟수'] / sector_stats['총예측횟수']) * 100
                sector_stats = sector_stats.sort_values(by='승률(%)', ascending=False).reset_index(drop=True)
                sector_stats.index = sector_stats.index + 1
                
                # 시각적으로 예쁘게 포맷팅
                sector_display = sector_stats.copy()
                sector_display['승률(%)'] = sector_display['승률(%)'].apply(lambda x: f"{x:.1f}%")
                st.dataframe(sector_display, use_container_width=True)
                
                st.markdown("---")
                
                # 상세 기록 보기 기능
                with st.expander("🔍 전체 종목 일자별 예측 및 실제 결과 상세 로그 보기"):
                    df_filtered_bt_disp = df_filtered_bt.copy()
                    df_filtered_bt_disp['예측수익률'] = df_filtered_bt_disp['예측수익률'].apply(lambda x: f"{x:.2f}%")
                    df_filtered_bt_disp['실제수익률'] = df_filtered_bt_disp['실제수익률'].apply(lambda x: f"{x:.2f}%")
                    df_filtered_bt_disp['결과'] = df_filtered_bt_disp['적중'].apply(lambda x: "✅ 적중" if x == 1 else "❌ 실패")
                    df_filtered_bt_disp = df_filtered_bt_disp[['날짜', '카테고리', 'ETF명', '예측수익률', '실제수익률', '결과']].sort_values(by=['날짜', '카테고리'], ascending=[False, True])
                    st.dataframe(df_filtered_bt_disp, use_container_width=True)
else:
    st.sidebar.error(f"❌ `{csv_filename}` 파일을 찾을 수 없습니다.")