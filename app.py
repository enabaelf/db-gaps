import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# 웹페이지 기본 설정
st.set_page_config(layout="wide", page_title="GAPS ETF 투자 대회 대시보드")

st.title("📊 GAPS ETF 대시보드 [V4 - 상세조회 기능 활성화]")
st.markdown("과거 10년 주가 기반 상승 확률 및 기댓값 최적화 원픽 추천 시스템")

def to_numeric(val):
    if pd.isna(val) or val == '-':
        return np.nan
    return float(str(val).replace('%', ''))

@st.cache_data(show_spinner="⏳ CSV 파일을 읽어 10년 치 주가 분석 중... (약 1분 소요)")
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
        
        # 두 개의 탭 생성
        tab1, tab2 = st.tabs(["🏆 전략별 카테고리 원픽", "🔍 종목 상세조회"])

        # [탭 1] 메인 대시보드 리포트
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

        # [탭 2] 종목 상세조회 기능 활성화
        with tab2:
            st.subheader("🔍 ETF 개별 종목 정밀 데이터 조회")
            if not df_analysis.empty:
                # 드롭다운 선택상자 생성
                etf_names = df_analysis['ETF명'].tolist()
                selected_name = st.selectbox("조회할 ETF 종목을 선택하세요:", etf_names)
                
                # 선택한 종목의 데이터 추출
                etf_info = df_analysis[df_analysis['ETF명'] == selected_name].iloc[0]
                
                # 기본 정보 요약 카드형 대시보드
                m1, m2, m3 = st.columns(3)
                m1.metric("종목 코드", etf_info['종목코드'])
                m2.metric("투자 전략 카테고리", etf_info['카테고리'])
                m3.metric("과거 데이터 분석 기간", etf_info['분석일수(샘플)'])
                
                st.markdown("---")
                st.markdown(f"### 📊 `{selected_name}` 시나리오별 통계")
                
                # 확률 및 수익률 상세 지표 비교
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.info("**🔹 기본 조건 (어제 상승 시)**")
                    st.write(f"• 오늘 또 오를 확률: **{etf_info['어제상승_오늘상승확률']}**")
                    st.write(f"• 오늘 평균 수익률: **{etf_info['어제상승_오늘평균수익률']}**")
                with c2:
                    st.success("**🔥 모멘텀 조건 (3일 연속 상승 시)**")
                    st.write(f"• 추가 상승 확률: **{etf_info['3일상승_추가상승확률']}**")
                    st.write(f"• 평균 추가 상승폭: **{etf_info['3일상승_평균추가상승폭']}**")
                with c3:
                    st.warning("**🧊 역발상 조건 (3일 연속 하락 시)**")
                    st.write(f"• 기술적 반등 확률: **{etf_info['3일하락_반등확률']}**")
                    st.write(f"• 평균 반등 상승폭: **{etf_info['3일하락_평균반등폭']}**")
                
                # 주가 흐름 시각화 차트 추가
                st.markdown("---")
                st.markdown("### 📈 최근 1년 주가 추이 추적")
                raw_ticker = etf_info['종목코드'].replace('A', '')
                try:
                    chart_start = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
                    df_chart = fdr.DataReader(raw_ticker, chart_start)[['Close']].rename(columns={'Close': '주가(원)'})
                    st.line_chart(df_chart, use_container_width=True)
                except:
                    st.error("⚠️ 해당 종목의 실시간 차트 데이터를 불러올 수 없습니다.")
else:
    st.sidebar.error(f"❌ `{csv_filename}` 파일을 찾을 수 없습니다.")