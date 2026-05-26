import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# 웹페이지 기본 설정
st.set_page_config(layout="wide", page_title="GAPS ETF 투자 대회 대시보드")

st.title("📊 GAPS ETF 대시보드 [V6 - 표준 가중치 스코어 모델]")
st.markdown("월가 표준 선형 가중치(WMA) 기반 단기 과매도/모멘텀 정밀 추천 시스템")

def to_numeric(val):
    if pd.isna(val) or val == '-':
        return np.nan
    return float(str(val).replace('%', ''))

@st.cache_data(ttl=21600, show_spinner="⏳ 가중치 스코어 모델로 10년 치 주가 분석 중... (약 1분 소요)")
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
            if len(df) < 15: continue

            df['Price_Change'] = df['Price'].pct_change()
            
            # 당일 방향성 정의 (상승 +1, 하락 -1)
            df['Dir'] = np.where(df['Price_Change'] > 0, 1, -1)
            
            # 월가 표준 4일 선형 가중 스코어 계산 (어제 40%, 2일전 30%, 3일전 20%, 4일전 10%)
            df['Weight_Score'] = (
                df['Dir'].shift(1) * 0.4 +
                df['Dir'].shift(2) * 0.3 +
                df['Dir'].shift(3) * 0.2 +
                df['Dir'].shift(4) * 0.1
            )
            df = df.dropna()

            # 1. 과매도 상태 (-0.6 이하) 일 때 다음 날 반등 확률 및 평균 반등폭
            oversold_days = df[df['Weight_Score'] <= -0.6]
            if len(oversold_days) > 0:
                prob_reb = (oversold_days['Price_Change'] > 0).mean() * 100
                ret_reb = oversold_days['Price_Change'].mean() * 100
            else:
                prob_reb, ret_reb = np.nan, np.nan

            # 2. 강한 모멘텀 상태 (+0.6 이상) 일 때 다음 날 추가 상승 확률 및 평균 상승폭
            momentum_days = df[df['Weight_Score'] >= 0.6]
            if len(momentum_days) > 0:
                prob_mom = (momentum_days['Price_Change'] > 0).mean() * 100
                ret_mom = momentum_days['Price_Change'].mean() * 100
            else:
                prob_mom, ret_mom = np.nan, np.nan

            # 3. 오늘 자 실제 계산된 스코어 (가장 최신 영업일 기준)
            current_score = df['Weight_Score'].iloc[-1]

            summary_results.append({
                '종목코드': 'A' + ticker, 'ETF명': info['name'], '카테고리': info['category'],
                '현재 가중치 스코어': round(current_score, 2),
                '과매도_반등확률': f"{prob_reb:.2f}%" if not np.isnan(prob_reb) else "-",
                '과매도_평균반등폭': f"{ret_reb:.2f}%" if not np.isnan(ret_reb) else "-",
                '모멘텀_추가상승확률': f"{prob_mom:.2f}%" if not np.isnan(prob_mom) else "-",
                '모멘텀_평균추가폭': f"{ret_mom:.2f}%" if not np.isnan(ret_mom) else "-",
                '분석일수(샘플)': f"{len(df)}일"
            })
        except: pass

    df_final = pd.DataFrame(summary_results)
    for col in ['과매도_반등확률', '과매도_평균반등폭', '모멘텀_추가상승확률', '모멘텀_평균추가폭']:
        df_final[col + '_숫자'] = df_final[col].apply(to_numeric)
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
        
        tab1, tab2, tab3 = st.tabs(["🏆 스코어 기반 전략 원픽", "🔍 종목 상세조회", "🔬 지난주 가중치 예측 검증"])

        # [탭 1] 메인 대시보드 리포트
        with tab1:
            st.markdown("### 💡 현재 가중치 스코어 해석 가이드")
            st.info("💡 **스코어가 -0.6 이하 이면?** 극단적 과매도 상태로 역사적 **기술적 반등** 타점입니다. \n\n🔥 **스코어가 +0.6 이상 이면?** 강력한 상승 정배열 상태로 **추격 매수(모멘텀)** 타점입니다.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🧊 [역발상 원픽] 현재 가장 과매도된 종목")
                # 스코어가 낮을수록 과매도 상태
                df_oversold_pick = df_analysis.sort_values(by='현재 가중치 스코어').head(5)
                st.dataframe(df_oversold_pick[['카테고리', '종목코드', 'ETF명', '현재 가중치 스코어', '과매도_반등확률', '과매도_평균반등폭']], use_container_width=True)
            with col2:
                st.subheader("🔥 [모멘텀 원픽] 현재 상승세가 가장 강한 종목")
                # 스코어가 높을수록 모멘텀 강함
                df_momentum_pick = df_analysis.sort_values(by='현재 가중치 스코어', ascending=False).head(5)
                st.dataframe(df_momentum_pick[['카테고리', '종목코드', 'ETF명', '현재 가중치 스코어', '모멘텀_추가상승확률', '모멘텀_평균추가폭']], use_container_width=True)
            
            st.subheader("📋 전체 ETF 실시간 스코어 및 통계 리포트")
            st.dataframe(df_analysis[['카테고리', '종목코드', 'ETF명', '현재 가중치 스코어', '과매도_반등확률', '과매도_평균반등폭', '모멘텀_추가상승확률', '모멘텀_평균추가폭', '분석일수(샘플)']], use_container_width=True)

        # [탭 2] 종목 상세조회
        with tab2:
            st.subheader("🔍 ETF 개별 종목 가중치 스코어 조회")
            if not df_analysis.empty:
                selected_name = st.selectbox("조회할 ETF 종목을 선택하세요:", df_analysis['ETF명'].tolist(), key="tab2_select")
                etf_info = df_analysis[df_analysis['ETF명'] == selected_name].iloc[0]
                
                m1, m2, m3 = st.columns(3)
                m1.metric("종목 코드", etf_info['종목코드'])
                m2.metric("현재 가중치 스코어", f"{etf_info['현재 가중치 스코어']}")
                m3.metric("과거 데이터 분석 기간", etf_info['분석일수(샘플)'])
                
                st.markdown("---")
                c1, c2 = st.columns(2)
                with c1:
                    st.warning("🧊 **역사적으로 이 종목의 스코어가 -0.6 이하로 내려갔을 때**")
                    st.write(f"• 다음 날 반등 성공 확률: **{etf_info['과매도_반등확률']}**")
                    st.write(f"• 반등 시 평균 기대 수익률: **{etf_info['과매도_평균반등폭']}**")
                with c2:
                    st.success("🔥 **역사적으로 이 종목의 스코어가 +0.6 이상으로 올라갔을 때**")
                    st.write(f"• 다음 날 추가 상승 확률: **{etf_info['모멘텀_추가상승확률']}**")
                    st.write(f"• 추가 상승 시 평균 기대 폭: **{etf_info['모멘텀_평균추가폭']}**")
                
                st.markdown("---")
                st.markdown("### 📈 최근 1년 주가 추이 추적")
                raw_ticker = etf_info['종목코드'].replace('A', '')
                try:
                    chart_start = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
                    df_chart = fdr.DataReader(raw_ticker, chart_start)[['Close']].rename(columns={'Close': '주가(원)'})
                    st.line_chart(df_chart, use_container_width=True)
                except:
                    st.error("⚠️ 차트 데이터를 불러올 수 없습니다.")

        # [탭 3] 지난주 데이터 예측력 검증
        with tab3:
            st.subheader("🔬 가중치 모델 아웃오브샘플 검증 (지난주 결과 대조)")
            if not df_analysis.empty:
                selected_name_bt = st.selectbox("검증할 ETF 종목을 선택하세요:", df_analysis['ETF명'].tolist(), key="tab3_select")
                bt_code = df_analysis[df_analysis['ETF명'] == selected_name_bt].iloc[0]['종목코드'].replace('A', '')
                
                try:
                    end_date = datetime.today().strftime('%Y-%m-%d')
                    start_date = (datetime.today() - timedelta(days=365*10)).strftime('%Y-%m-%d')
                    df_bt = fdr.DataReader(bt_code, start_date, end_date)[['Close']].rename(columns={'Close': 'Price'})
                    
                    df_bt['Price_Change'] = df_bt['Price'].pct_change()
                    df_bt['Dir'] = np.where(df_bt['Price_Change'] > 0, 1, -1)
                    df_bt['Weight_Score'] = (
                        df_bt['Dir'].shift(1) * 0.4 +
                        df_bt['Dir'].shift(2) * 0.3 +
                        df_bt['Dir'].shift(3) * 0.2 +
                        df_bt['Dir'].shift(4) * 0.1
                    )
                    df_bt = df_bt.dropna()
                    
                    df_past_10y = df_bt.iloc[:-5]
                    df_last_week = df_bt.tail(5)
                    
                    # 과거 가중치 통계 기준 확률 재계산
                    past_oversold = df_past_10y[df_past_10y['Weight_Score'] <= -0.6]
                    past_prob_reb = (past_oversold['Price_Change'] > 0).mean() * 100 if len(past_oversold) > 0 else np.nan
                    
                    past_momentum = df_past_10y[df_past_10y['Weight_Score'] >= 0.6]
                    past_prob_mom = (past_momentum['Price_Change'] > 0).mean() * 100 if len(past_momentum) > 0 else np.nan
                    
                    st.markdown("#### 📅 실제 지난주(5영업일) 스코어 예측 결과 매칭")
                    
                    bt_records = []
                    for date, row in df_last_week.iterrows():
                        date_str = date.strftime('%Y-%m-%d')
                        actual_return = f"{row['Price_Change']*100:.2f}%"
                        actual_result = "🔺 상승" if row['Dir'] == 1 else "🔻 하락"
                        
                        score_now = round(row['Weight_Score'], 2)
                        condition = f"일반 상태 (Score: {score_now})"
                        hit = "-"
                        
                        if score_now <= -0.6:
                            condition = f"🧊 과매도 구간 (Score: {score_now})"
                            hit = "✅ 적중" if row['Dir'] == 1 else "❌ 실패"
                        elif score_now >= 0.6:
                            condition = f"🔥 모멘텀 구간 (Score: {score_now})"
                            hit = "✅ 적중" if row['Dir'] == 1 else "❌ 실패"
                            
                        bt_records.append({
                            "날짜": date_str,
                            "그 전날까지 누적된 가중치 상태": condition,
                            "실제 결과": actual_result,
                            "당일 수익률": actual_return,
                            "예측 적중 여부(상승예측 기준)": hit
                        })
                        
                    st.dataframe(pd.DataFrame(bt_records), use_container_width=True)
                except Exception as e:
                    st.error(f"⚠️ 백테스팅 계산 오류: {e}")
else:
    st.sidebar.error(f"❌ `{csv_filename}` 파일을 찾을 수 없습니다.")