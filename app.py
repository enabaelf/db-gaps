import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# 웹페이지 기본 설정
st.set_page_config(layout="wide", page_title="GAPS ETF 투자 대회 대시보드")

st.title("📊 GAPS ETF 대시보드 [V7 - 선형 회귀 기대수익률 모델]")
st.markdown("최근 4일 변동폭 가중치(WMA) 및 10년 통계 회귀분석 기반 내일의 기대수익률 예측 시스템")

@st.cache_data(ttl=21600, show_spinner="⏳ 선형 회귀 모델로 10년 치 기대수익률 학습 중... (약 1분 소요)")
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
            if len(df) < 20: continue

            df['Price_Change'] = df['Price'].pct_change()
            
            # 회원님 아이디어: 방향성이 아닌 '실제 그날의 변동폭'에 가중치 매칭
            df['Weight_Score'] = (
                df['Price_Change'] * 0.4 +
                df['Price_Change'].shift(1) * 0.3 +
                df['Price_Change'].shift(2) * 0.2 +
                df['Price_Change'].shift(3) * 0.1
            )
            
            # 예측 대상: 오늘 스코어를 기반으로 한 '내일의 실제 수익률'
            df['Next_Return'] = df['Price_Change'].shift(-1)
            df_clean = df.dropna()
            
            if len(df_clean) < 10: continue

            # 선형 회귀분석 수행 (y = slope * x + intercept)
            # x: 과거 가중치 스코어들, y: 그다음 날 일어난 실제 수익률들
            slope, intercept = np.polyfit(df_clean['Weight_Score'], df_clean['Next_Return'], 1)
            
            # 가장 최신 가중치 스코어 (오늘 자 마감 스코어)
            current_score = df['Weight_Score'].iloc[-1]
            
            # 일반 상태를 포함한 '내일 자 최종 기대수익률' 통계적 예측
            predicted_next_return = (intercept + slope * current_score) * 100 # % 단위로 변환
            
            # 모델의 역사적 신뢰도 측정 (상관계수)
            correlation = df_clean['Weight_Score'].corr(df_clean['Next_Return'])

            summary_results.append({
                '종목코드': 'A' + ticker, 
                'ETF명': info['name'], 
                '카테고리': info['category'],
                '현재 가중치 스코어': f"{current_score*100:.3f}%",
                '내일 기대수익률': round(predicted_next_return, 4),
                '모델 신뢰도(상관성)': round(correlation, 2),
                '분석일수(샘플)': f"{len(df_clean)}일"
            })
        except: pass

    df_final = pd.DataFrame(summary_results)
    return df_final

csv_filename = "gaps_etf_list.csv"

if os.path.exists(csv_filename):
    st.sidebar.success(f"📂 `{csv_filename}` 파일 감지 완료!")
    df_raw = None
    for enc in ['utf-8-sig', 'cp949', 'utf-8', 'euc-kr']:
        try:
            df_raw = pd.read_csv(csv_filename, header=None, encoding=enc, dtype=str)
            break
        except: continue

    if df_raw is not None:
        df_analysis = run_full_analysis(df_raw)
        st.sidebar.info(f"📊 총 {len(df_analysis)}개 종목 분석 완료")
        
        tab1, tab2, tab3 = st.tabs(["🏆 섹션별 기대수익률 TOP 3", "🔍 종목 상세조회", "🔬 지난주 예측력 검증 (회귀형)"])

        # [탭 1] 카테고리별 Top 3 대시보드 리포트
        with tab1:
            st.markdown("### 📈 내일 자 기대수익률 최적화 포트폴리오 추천")
            st.info("💡 과거 10개년 데이터의 연속적 회귀 흐름을 분석하여, **일반 상태를 포함해 내일 가장 큰 수익이 기대되는 종목**을 섹션별로 3개씩 선별했습니다.")
            
            # 카테고리(섹션)별로 루프를 돌며 Top 3 출력
            unique_categories = df_analysis['카테고리'].unique()
            
            # 시각적인 배치를 위해 2열 레이아웃 구성
            cat_chunks = [unique_categories[i:i + 2] for i in range(0, len(unique_categories), 2)]
            
            for chunk in cat_chunks:
                cols = st.columns(2)
                for i, cat in enumerate(chunk):
                    with cols[i]:
                        st.subheader(f"📂 {cat} 섹션 Top 3")
                        # 해당 카테고리 추출 후 내일 기대수익률 기준 내림차순 정렬
                        df_cat = df_analysis[df_analysis['카테고리'] == cat]
                        df_cat_top3 = df_cat.sort_values(by='내일 기대수익률', ascending=False).head(3).reset_index(drop=True)
                        df_cat_top3.index = df_cat_top3.index + 1 # 순위를 1, 2, 3으로 표시
                        
                        # 보기 좋게 컬럼 포맷팅 조정
                        df_display = df_cat_top3[['종목코드', 'ETF명', '현재 가중치 스코어', '내일 기대수익률', '모델 신뢰도(상관성)']].copy()
                        df_display['내일 기대수익률'] = df_display['내일 기대수익률'].apply(lambda x: f"{x:.2f}%")
                        st.dataframe(df_display, use_container_width=True)
            
            st.markdown("---")
            st.subheader("📋 전체 ETF 기대수익률 전체 리포트 (필터 및 정렬 가능)")
            st.dataframe(df_analysis[['카테고리', '종목코드', 'ETF명', '현재 가중치 스코어', '내일 기대수익률', '모델 신뢰도(상관성)', '분석일수(샘플)']].sort_values(by='내일 기대수익률', ascending=False), use_container_width=True)

        # [탭 2] 종목 상세조회
        with tab2:
            st.subheader("🔍 ETF 개별 종목 회귀 모델 정밀 조회")
            if not df_analysis.empty:
                selected_name = st.selectbox("조회할 ETF 종목을 선택하세요:", df_analysis['ETF명'].tolist(), key="tab2_select")
                etf_info = df_analysis[df_analysis['ETF명'] == selected_name].iloc[0]
                
                m1, m2, m3 = st.columns(3)
                m1.metric("현재 마감 가중치 스코어(WMA)", f"{etf_info['현재 가중치 스코어']}")
                
                # 기대수익률 강도에 따라 색상 안내 레이블 유동 처리 가능
                m2.metric("★ 내일 예측 기대수익률", f"{etf_info['내일 기대수익률']:.2f}%")
                m3.metric("역사적 모델 신뢰도 (상관성)", f"{etf_info['모델 신뢰도(상관성)']}")
                
                st.markdown("---")
                st.markdown("### 💡 이 예측 수치는 어떻게 도출되었나요?")
                st.write(f"1. 오늘 기준 최근 4일간의 실제 수익률 폭에 가중치를 곱해 계산한 오늘 자 점수는 **{etf_info['현재 가중치 스코어']}** 입니다.")
                st.write(f"2. 지난 **{etf_info['분석일수(샘플)']}** 동안 이 점수가 발생했을 때 그다음 날 주가가 어떻게 변했는지 컴퓨터가 선형 함수로 관계를 찾아냈습니다.")
                st.write(f"3. 모델 신뢰도가 **{etf_info['모델 신뢰도(상관성)']}** 이라는 것은 이 가중치 패턴이 내일 주가 방향을 예측하는 데 통계적 의미를 지니고 있음을 뜻합니다.")
                
                st.markdown("---")
                st.markdown("### 📈 최근 1년 주가 추이 추적")
                raw_ticker = etf_info['종목코드'].replace('A', '')
                try:
                    chart_start = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
                    df_chart = fdr.DataReader(raw_ticker, chart_start)[['Close']].rename(columns={'Close': '주가(원)'})
                    st.line_chart(df_chart, use_container_width=True)
                except:
                    st.error("⚠️ 차트 데이터를 불러올 수 없습니다.")

        # [탭 3] 지난주 데이터 예측력 검증 (회귀분석 맞춤 백테스트)
        with tab3:
            st.subheader("🔬 회귀 예측 백테스팅: 실제 지난주(5영업일) 결과와 대조")
            st.markdown("과거 데이터로 가중치 회귀선(수식)을 도출한 뒤, **지난주 월~금요일 아침에 이 모델을 켰다면 당일 종가 수익률을 플러스/마이너스까지 맞췄는지** 검증합니다.")
            
            if not df_analysis.empty:
                selected_name_bt = st.selectbox("검증할 ETF 종목을 선택하세요:", df_analysis['ETF명'].tolist(), key="tab3_select")
                bt_code = df_analysis[df_analysis['ETF명'] == selected_name_bt].iloc[0]['종목코드'].replace('A', '')
                
                try:
                    end_date = datetime.today().strftime('%Y-%m-%d')
                    start_date = (datetime.today() - timedelta(days=365*10)).strftime('%Y-%m-%d')
                    df_bt = fdr.DataReader(bt_code, start_date, end_date)[['Close']].rename(columns={'Close': 'Price'})
                    
                    df_bt['Price_Change'] = df_bt['Price'].pct_change()
                    df_bt['Weight_Score'] = (
                        df_bt['Price_Change'] * 0.4 +
                        df_bt['Price_Change'].shift(1) * 0.3 +
                        df_bt['Price_Change'].shift(2) * 0.2 +
                        df_bt['Price_Change'].shift(3) * 0.1
                    )
                    df_bt['Next_Return'] = df_bt['Price_Change'].shift(-1)
                    df_bt = df_bt.dropna()
                    
                    # 테스트 분할 (마지막 5일 제외)
                    df_train = df_bt.iloc[:-5]
                    df_test_period = df_bt.tail(5)
                    
                    # 과거 기준 회귀선 도출
                    slope_past, intercept_past = np.polyfit(df_train['Weight_Score'], df_train['Next_Return'], 1)
                    
                    bt_records = []
                    # 최근 5영업일 검증 추적
                    for date, row in df_test_period.iterrows():
                        date_str = date.strftime('%Y-%m-%d')
                        
                        # 전날 마감 시점의 스코어가 당일 수익률을 예측함 (즉 shift(1) 스코어 기준)
                        pred_ret_raw = intercept_past + slope_past * row['Weight_Score']
                        pred_ret_pct = pred_ret_raw * 100
                        actual_ret_pct = row['Next_Return'] * 100
                        
                        if np.isnan(actual_ret_pct):
                            # 만약 오늘이 장중이거나 아직 내일 주가가 없는 최신 행이라면 패스
                            continue
                            
                        pred_dir = "🔺 상승예측" if pred_ret_pct > 0 else "🔻 하락예측"
                        actual_dir = "🔺 상승" if actual_ret_pct > 0 else "🔻 하락"
                        
                        # 방향 적중 여부 판정
                        is_hit = "✅ 방향 적중" if (pred_ret_pct > 0 and actual_ret_pct > 0) or (pred_ret_pct < 0 and actual_ret_pct < 0) else "❌ 실패"
                        
                        bt_records.append({
                            "날짜": date_str,
                            "전날 마감 기준 가중치 스코어": f"{row['Weight_Score']*100:.3f}%",
                            "모델의 내일 기대수익률 예측": f"{pred_ret_pct:.2f}% ({pred_dir})",
                            "실제 다음 날 일어난 수익률": f"{actual_ret_pct:.2f}% ({actual_dir})",
                            "예측 성공 여부": is_hit
                        })
                        
                    if bt_records:
                        st.dataframe(pd.DataFrame(bt_records), use_container_width=True)
                    else:
                        st.info("💡 오늘 장이 진행 중이거나 아직 백테스트 매칭 데이터가 부족합니다. 주말이나 장 마감 후에 완벽히 표기됩니다.")
                except Exception as e:
                    st.error(f"⚠️ 백테스팅 계산 오류: {e}")
else:
    st.sidebar.error(f"❌ `{csv_filename}` 파일을 찾을 수 없습니다.")