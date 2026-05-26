import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# 웹페이지 기본 설정
st.set_page_config(layout="wide", page_title="GAPS ETF 투자 대회 대시보드")

st.title("📊 GAPS ETF 대시보드 [V9 - 거래량·추세 및 성과 비교 완료]")
st.markdown("최근 4일 거래량 가중 변동폭 및 20일 이평선 결합 모델 + 지난주 실제 수익률 리얼 백테스팅")

@st.cache_data(ttl=21600, show_spinner="⏳ 거래량·추세 분석 및 지난주 모의투자 시뮬레이션 중... (약 1분 소요)")
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
            # 주가와 함께 거래량(Volume) 데이터도 함께 로드
            df = fdr.DataReader(ticker, start_date, end_date)[['Close', 'Volume']].rename(columns={'Close': 'Price'})
            if len(df) < 40: continue

            df['Price_Change'] = df['Price'].pct_change()
            
            # [개선 2] 거래량 지표 추가: 20일 평균 거래량 대비 오늘 거래량 비율 계산
            df['Vol_MA20'] = df['Volume'].rolling(20).mean()
            df['Vol_Ratio'] = df['Volume'] / df['Vol_MA20']
            df['Vol_Ratio'] = df['Vol_Ratio'].fillna(1.0).replace([np.inf, -np.inf], 1.0)
            df['Vol_Weight'] = np.clip(df['Vol_Ratio'], 0.5, 3.0) # 극단값 억제 클리핑
            
            # 거래량이 터진 날의 변동폭에 더 큰 가중치를 주는 변동폭 생성
            df['VW_Change'] = df['Price_Change'] * df['Vol_Weight']
            
            # 선형 가중치 스코어 산출
            df['Weight_Score'] = (
                df['VW_Change'] * 0.4 +
                df['VW_Change'].shift(1) * 0.3 +
                df['VW_Change'].shift(2) * 0.2 +
                df['VW_Change'].shift(3) * 0.1
            )
            
            # [개선 3] 20일 이동평균선 장기 추세 필터 정의
            df['MA20'] = df['Price'].rolling(20).mean()
            df['Trend'] = np.where(df['Price'] >= df['MA20'], "🟢상승세", "🔴하락세")
            
            df['Next_Return'] = df['Price_Change'].shift(-1)
            df_clean = df.dropna(subset=['Weight_Score'])
            
            if len(df_clean) < 25: continue

            # --- 1. 오늘 자 마감 기준 내일 기대수익률 예측 (전체 데이터 이용) ---
            df_for_pred = df_clean.dropna(subset=['Next_Return'])
            slope_all, intercept_all = np.polyfit(df_for_pred['Weight_Score'], df_for_pred['Next_Return'], 1)
            
            current_score = df_clean['Weight_Score'].iloc[-1]
            predicted_next_return = (intercept_all + slope_all * current_score) * 100
            correlation = df_for_pred['Weight_Score'].corr(df_for_pred['Next_Return'])
            current_trend = df_clean['Trend'].iloc[-1]

            # --- 2. [신설] 5영업일 전(지난주 시작점) 모의투자 백테스팅 데이터 산출 ---
            # 지난주 5일간의 실제 누적 수익률 계산
            actual_5d_return = (df['Price'].iloc[-1] / df['Price'].iloc[-5] - 1) * 100
            
            # 5영업일 전 시점까지의 데이터만 잘라서 회귀선 도출 (미래 데이터 오염 방지)
            df_past = df_clean.iloc[:-5]
            df_past_clean = df_past.dropna(subset=['Next_Return'])
            
            if len(df_past_clean) > 15:
                slope_5d, intercept_5d = np.polyfit(df_past_clean['Weight_Score'], df_past_clean['Next_Return'], 1)
                # 5영업일 전 날짜의 가중치 점수를 대입하여 '당시 모델의 내일 기대수익률 예측치' 계산
                pred_return_5d = (intercept_5d + slope_5d * df_clean['Weight_Score'].iloc[-5]) * 100
            else:
                pred_return_5d = np.nan

            summary_results.append({
                '종목코드': 'A' + ticker, 
                'ETF명': info['name'], 
                '카테고리': info['category'],
                '현재 장기추세(20MA)': current_trend,
                '현재 가중치 스코어': f"{current_score*100:.2f}%",
                '내일 기대수익률': round(predicted_next_return, 4),
                '모델 신뢰도': round(correlation, 2),
                '지난주 모델 예측치(5d전)': pred_return_5d,
                '지난주 실제 누적수익률': actual_5d_return
            })
        except: pass

    df_final = pd.DataFrame(summary_results)
    return df_final

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
        st.sidebar.success(f"📂 분석 완료! (종목: {len(df_analysis)}개)")
        
        tab1, tab2, tab3 = st.tabs(["🏆 섹션별 기대수익률 TOP 3", "🌲 [숲 보기] 지난주 모의투자 성과 비교", "🔍 종목 상세조회"])

        # [탭 1] 메인 화면
        with tab1:
            st.markdown("### 📈 내일 자 기대수익률 최적화 포트폴리오 추천")
            st.caption("개선 포인트: 거래량이 동반된 주가 변동폭에 가중치를 부여하고 20일 이동평균선(장기추세) 상태를 반영했습니다.")
            
            unique_categories = df_analysis['카테고리'].unique()
            cat_chunks = [unique_categories[i:i + 2] for i in range(0, len(unique_categories), 2)]
            
            for chunk in cat_chunks:
                cols = st.columns(2)
                for i, cat in enumerate(chunk):
                    with cols[i]:
                        st.subheader(f"📂 {cat} 섹션 Top 3")
                        df_cat = df_analysis[df_analysis['카테고리'] == cat]
                        df_cat_top3 = df_cat.sort_values(by='내일 기대수익률', ascending=False).head(3).reset_index(drop=True)
                        df_cat_top3.index = df_cat_top3.index + 1
                        
                        df_display = df_cat_top3[['ETF명', '현재 장기추세(20MA)', '현재 가중치 스코어', '내일 기대수익률']].copy()
                        df_display['내일 기대수익률'] = df_display['내일 기대수익률'].apply(lambda x: f"{x:.2f}%")
                        st.dataframe(df_display, use_container_width=True)

        # [탭 2] 숲 보기: 지난주 모의투자 성과 비교 (회원님 핵심 요청 사항)
        with tab2:
            st.subheader("🌲 [숲 보기] 지난주 모델 예측 포트폴리오 vs 시장 최고 수익률 비교")
            st.markdown("과거 섹터 통계와 무관하게, 오직 **'최종 계좌 수익률 극대화'** 관점에서 지난 일주일(5영업일)간의 모델 성적을 복기합니다.")
            
            if not df_analysis.empty:
                # 유효한 백테스트 샘플 추출
                df_bt = df_analysis.dropna(subset=['지난주 모델 예측치(5d전)', '지난주 실제 누적수익률'])
                
                # 1. 5일 전 시점에 모델이 전체 시장에서 가장 유망하다고 찍었던 탑 5 종목 (섹터 통합)
                df_model_picks = df_bt.sort_values(by='지난주 모델 예측치(5d전)', ascending=False).head(5).reset_index(drop=True)
                df_model_picks.index = df_model_picks.index + 1
                
                # 2. 지난 5일 동안 실제로 전체 시장에서 가장 높은 누적 수익률을 기록한 진짜 정답 탑 5 종목
                df_market_winners = df_bt.sort_values(by='지난주 실제 누적수익률', ascending=False).head(5).reset_index(drop=True)
                df_market_winners.index = df_market_winners.index + 1
                
                # 핵심 요약 지표 (Metric) 출력
                avg_model_return = df_model_picks['지난주 실제 누적수익률'].mean()
                max_market_return = df_market_winners['지난주 실제 누적수익률'].mean()
                avg_market_return = df_bt['지난주 실제 누적수익률'].mean()
                
                m1, m2, m3 = st.columns(3)
                m1.metric("🤖 지난주 모델 추천포트폴리오 평균수익률", f"{avg_model_return:.2f}%")
                m2.metric("👑 지난주 시장 최고존엄(Top5) 평균수익률", f"{max_market_return:.2f}%")
                m3.metric("📊 전체 ETF 유니버스 전체 평균수익률", f"{avg_market_return:.2f}%")
                
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### 🤖 5영업일 전 우리 모델이 베팅했던 종목군과 실제 성과")
                    st.caption("지난주 월요일 아침 기준, 모델 예측치가 가장 높았던 상위 5개 종목의 실제 일주일 누적 성적표입니다.")
                    df_model_disp = df_model_picks[['카테고리', 'ETF명', '지난주 모델 예측치(5d전)', '지난주 실제 누적수익률']].copy()
                    df_model_disp['지난주 모델 예측치(5d전)'] = df_model_disp['지난주 모델 예측치(5d전)'].apply(lambda x: f"{x:.3f}%")
                    df_model_disp['지난주 실제 누적수익률'] = df_model_disp['지난주 실제 누적수익률'].apply(lambda x: f"{x:.2f}%")
                    st.dataframe(df_model_disp, use_container_width=True)
                    
                with col2:
                    st.markdown("#### 👑 지난 일주일 동안 실제 최고 수익률을 낸 종목군 (정답지)")
                    st.caption("섹터 불문하고 지난주 시장을 완벽하게 씹어먹은 실제 누적수익률 상위 5개 종목입니다.")
                    df_winner_disp = df_market_winners[['카테고리', 'ETF명', '지난주 실제 누적수익률', '지난주 모델 예측치(5d전)']].copy()
                    df_winner_disp['지난주 실제 누적수익률'] = df_winner_disp['지난주 실제 누적수익률'].apply(lambda x: f"{x:.2f}%")
                    df_winner_disp['지난주 모델 예측치(5d전)'] = df_winner_disp['지난주 모델 예측치(5d전)'].apply(lambda x: f"{x:.3f}%" if not pd.isna(x) else "-")
                    st.dataframe(df_winner_disp, use_container_width=True)
                    
                st.info(f"💡 **숲 보기 피드백**: 모델 픽 평균 수익률({avg_model_return:.2f}%)이 시장 전체 평균({avg_market_return:.2f}%)보다 높다면, 우리 모델은 시장보다 우월한 자산 배분 능력을 입증한 것입니다. 이제 우측 정답지 종목들과 모델 픽 간의 교집합을 넓히는 방향으로 가중치를 다듬어가면 됩니다.")

        # [탭 3] 상세 조회
        with tab2:
            st.subheader("🔍 ETF 개별 종목 회귀 모델 정밀 조회")
            if not df_analysis.empty:
                selected_name = st.selectbox("조회할 ETF 종목을 선택하세요:", df_analysis['ETF명'].tolist(), key="tab3_select")
                etf_info = df_analysis[df_analysis['ETF명'] == selected_name].iloc[0]
                
                m1, m2, m3 = st.columns(3)
                m1.metric("현재 마감 가중치 스코어(WMA)", f"{etf_info['현재 가중치 스코어']}")
                m2.metric("★ 내일 예측 기대수익률", f"{etf_info['내일 기대수익률']:.2f}%")
                m3.metric("모델 신뢰도 (상관성)", f"{etf_info['모델 신뢰도']}")
else:
    st.sidebar.error(f"❌ `{csv_filename}` 파일을 찾을 수 없습니다.")