import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# 웹페이지 기본 설정
st.set_page_config(layout="wide", page_title="GAPS ETF 투자 대회 대시보드")

st.title("🥇 GAPS ETF 대시보드 [V33 - 최신 장 마감 데이터 완벽 반영 모델]")
st.markdown("💡 **구조적 버그 수정 완료:** 당일 종가가 연산 과정에서 누락되던 shift 공식을 전면 전향 수정하여, 오늘 마감된 데이터가 실시간 백테스트와 내일 기댓값 계산에 완벽히 반영됩니다.")

# --- 캐시 강제 초기화 버튼 ---
st.sidebar.header("🔄 데이터 동기화")
if st.sidebar.button("🔄 최신 데이터 강제 갱신", type="primary", use_container_width=True):
    st.cache_data.clear()  
    st.sidebar.success("⏳ 캐시가 초기화되었습니다! 최신 데이터를 다시 불러옵니다...")
    st.rerun()

# --- 수동 기록장용 파일 초기화 로직 ---
log_filename = "actual_trade_history.csv"
if not os.path.exists(log_filename):
    df_log_init = pd.DataFrame(columns=['매매일자', '나의 실제 계좌수익률(%)', '당일 주력 매수 종목', '투자 복기 및 메모'])
    df_log_init.loc[0] = [(datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d'), 0.0, "현금 보유", "기록장 시동! 여기에 어제 기록을 적으세요."]
    df_log_init.to_csv(log_filename, index=False, encoding='utf-8-sig')

# --- 좌측 사이드바 설정 ---
st.sidebar.header("⚙️ 앙상블 모델 설정")
train_window_option = st.sidebar.selectbox(
    "최근 장세 반영 기간 (회귀 가중치용)",
    ["500 영업일 (약 2년)", "700 영업일 (약 2.8년)", "1000 영업일 (약 4년)"],
    index=0
)

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
        if row[target_col] <= 0: break 
        
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
        elif '단기' in raw_cat or '금리' in raw_cat or '초단기' in raw_cat: limit, asset_type, c_name = 50, '안전', '단기채권'
        elif '채권' in raw_cat: limit, asset_type, c_name = 50, '안전', '기타_안전채권'
        elif '주식' in raw_cat: limit, asset_type, c_name = 10, '위험', '기타_위험주식'
        
        current_cat_alloc = cat_alloc.get(c_name, 0.0)
        available_cat = limit - current_cat_alloc
        
        weight = min(total_budget, available_cat, 20.0)
        if asset_type == '위험':
            weight = min(weight, risk_budget)
            
        if weight > 0:
            item = {
                'ETF명': row['ETF명'], '카테고리': c_name, '자산군': asset_type,
                '추천비중(%)': weight, '앙상블 기댓값(%)': row[target_col]
            }
            if has_actual: item['실제수익률(%)'] = row['Actual']
            portfolio.append(item)
            
            total_budget -= weight
            if asset_type == '위험': risk_budget -= weight
            cat_alloc[c_name] = current_cat_alloc + weight

    if total_budget > 0:
        portfolio.append({
            'ETF명': '현금보유 (Cash)', '카테고리': '현금', '자산군': '안전',
            '추천비중(%)': total_budget, '앙상블 기댓값(%)': 0.0, '실제수익률(%)': 0.0 if has_actual else np.nan
        })
        
    return pd.DataFrame(portfolio)

@st.cache_data(ttl=21600, show_spinner="⏳ 10년 패턴 추출 및 최신 종가 데이터 반영 중... (약 15초 소요)")
def run_full_analysis(df_raw, train_window_option):
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
    
    # 10년치 데이터를 수집하되, end_date를 생략하여 제공 서버가 가진 완전한 최신 데이터를 제한없이 가져옴
    start_date = (datetime.today() - timedelta(days=365*10)).strftime('%Y-%m-%d')

    if "1000" in train_window_option: fit_window = 1000
    elif "700" in train_window_option: fit_window = 700
    else: fit_window = 500

    actual_latest_date = "데이터 없음"

    for ticker, info in ticker_dict.items():
        try:
            df = fdr.DataReader(ticker, start_date)[['Close']].rename(columns={'Close': 'Price'})
            if len(df) < 40: continue

            actual_latest_date = df.index[-1].strftime('%Y-%m-%d')

            df['Price_Change'] = df['Price'].pct_change()
            df['Dir'] = np.where(df['Price_Change'] > 0, 1, 0)
            
            # [버그 수정의 핵심] Target을 밀지 않고 과거 Feature 패턴들을 뒤로 Shift하여 데이터 일치화
            df['Target_Return'] = df['Price_Change']  # 당일 수익률 자체를 타겟화
            df['L1'] = df['Dir'].shift(1)             # 1영업일 전 방향
            df['L2'] = df['Dir'].shift(2)             # 2영업일 전 방향
            df['L3'] = df['Dir'].shift(3)             # 3영업일 전 방향
            df['L4'] = df['Dir'].shift(4)             # 4영업일 전 방향
            
            df_clean = df.dropna(subset=['L4', 'Target_Return']).copy()
            if len(df_clean) < 100: continue
            
            annual_std_dev = df_clean['Price_Change'].std() * np.sqrt(252) * 100
            
            e1_map = df_clean.groupby('L1')['Target_Return'].mean().to_dict()
            e2_map = df_clean.groupby(['L1', 'L2'])['Target_Return'].mean().to_dict()
            e3_map = df_clean.groupby(['L1', 'L2', 'L3'])['Target_Return'].mean().to_dict()
            e4_map = df_clean.groupby(['L1', 'L2', 'L3', 'L4'])['Target_Return'].mean().to_dict()
            
            df_clean['E1'] = df_clean['L1'].map(e1_map).fillna(0.0)
            df_clean['E2'] = df_clean.set_index(['L1', 'L2']).index.map(e2_map.get).fillna(0.0)
            df_clean['E3'] = df_clean.set_index(['L1', 'L2', 'L3']).index.map(e3_map.get).fillna(0.0)
            df_clean['E4'] = df_clean.set_index(['L1', 'L2', 'L3', 'L4']).index.map(e4_map.get).fillna(0.0)
            
            actual_w = min(len(df_clean), fit_window)
            df_fit = df_clean.tail(actual_w)
            
            X_fit = df_fit[['E1', 'E2', 'E3', 'E4']].values
            y_fit = df_fit['Target_Return'].values
            X_design = np.column_stack((np.ones(len(X_fit)), X_fit))
            
            coeffs, _, _, _ = np.linalg.lstsq(X_design, y_fit, rcond=None)
            best_intercept = coeffs[0]
            best_w1, best_w2, best_w3, best_w4 = coeffs[1:]
            
            df_clean['Final_Pred'] = (best_intercept + df_clean['E1']*best_w1 + df_clean['E2']*best_w2 + df_clean['E3']*best_w3 + df_clean['E4']*best_w4) * 100
            df_clean['Actual'] = df_clean['Target_Return'] * 100
            
            final_correlation = df_clean['Final_Pred'].corr(df_clean['Actual'])
            if pd.isna(final_correlation): final_correlation = 0.0
            
            # 오늘 장 마감 데이터(6월 1일)까지 누락 없이 일지 기록 확보
            for date, row in df_clean.tail(60).iterrows():
                daily_records.append({
                    'Date': date.strftime('%Y-%m-%d'), 'ETF명': info['name'], '카테고리': info['category'],
                    'Pred': row['Final_Pred'], 'Actual': row['Actual']
                })

            # 오늘 장 마감 데이터를 기준으로 한 '완전한 내일 장세(Next Day)' 예측 기댓값 도출
            next_L1 = df['Dir'].iloc[-1]
            next_L2 = df['Dir'].iloc[-2]
            next_L3 = df['Dir'].iloc[-3]
            next_L4 = df['Dir'].iloc[-4]
            
            next_E1 = e1_map.get(next_L1, 0.0)
            next_E2 = e2_map.get((next_L1, next_L2), 0.0)
            next_E3 = e3_map.get((next_L1, next_L2, next_L3), 0.0)
            next_E4 = e4_map.get((next_L1, next_L2, next_L3, next_L4), 0.0)
            
            pred_tomorrow = (best_intercept + next_E1*best_w1 + next_E2*best_w2 + next_E3*best_w3 + next_E4*best_w4) * 100

            df['MA20'] = df['Price'].rolling(20).mean()
            df['Trend'] = np.where(df['Price'] >= df['MA20'], "🟢상승세", "🔴하락세")

            summary_results.append({
                '종목코드': 'A' + ticker, 'ETF명': info['name'], '카테고리': info['category'],
                '현재추세': df['Trend'].iloc[-1], '앙상블 기댓값(%)': pred_tomorrow,
                '연간 표준편차(%)': annual_std_dev, '모델 상관계수': final_correlation
            })
        except: pass
        
    return pd.DataFrame(summary_results), pd.DataFrame(daily_records), actual_latest_date

csv_filename = "gaps_etf_list.csv"

if os.path.exists(csv_filename):
    df_raw = None
    for enc in ['utf-8-sig', 'cp949', 'utf-8', 'euc-kr']:
        try:
            df_raw = pd.read_csv(csv_filename, header=None, encoding=enc, dtype=str)
            break
        except: continue

    if df_raw is not None:
        df_analysis, df_daily, latest_market_date = run_full_analysis(df_raw, train_window_option)
        
        st.sidebar.markdown("---")
        st.sidebar.info(f"📅 **서버 수집 완료 최신 데이터 날짜:**\n`{latest_market_date}`")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 오늘의 실전 매수 비중", 
            "🏆 섹터별 기댓값 TOP 3", 
            "🔥 캘린더 연동 백테스트", 
            "🔍 개별 종목 분석",
            "📝 나의 실전 매매 기록장"
        ])

        with tab1:
            st.subheader("🎯 오늘 자 최적화 포트폴리오 비중")
            df_pred_today = df_analysis.rename(columns={'앙상블 기댓값(%)': 'Pred'})
            optimal_portfolio = optimize_portfolio(df_pred_today, target_col='Pred')
            
            risk_sum = float(optimal_portfolio[optimal_portfolio['자산군'] == '위험']['추천비중(%)'].sum())
            safe_sum = float(optimal_portfolio[optimal_portfolio['자산군'] == '안전']['추천비중(%)'].sum())
            
            m1, m2 = st.columns(2)
            m1.metric("🔴 위험자산 총합 (Max 70%)", f"{risk_sum:.1f}%")
            m2.metric("🟢 안전자산 총합", f"{safe_sum:.1f}%")
            
            st.dataframe(optimal_portfolio[['자산군', '카테고리', 'ETF명', '추천비중(%)', '앙상블 기댓값(%)']].style.format({'추천비중(%)': '{:.1f}%', '앙상블 기댓값(%)': '{:.3f}%'}), use_container_width=True)

        with tab2:
            st.subheader("🏆 카테고리별 앙상블 기댓값 TOP 3")
            unique_cats = df_analysis['카테고리'].unique()
            for i in range(0, len(unique_cats), 2):
                cols = st.columns(2)
                for j, cat in enumerate(unique_cats[i:i+2]):
                    with cols[j]:
                        st.markdown(f"#### 📂 {cat}")
                        df_cat = df_analysis[df_analysis['카테고리'] == cat].sort_values(by='앙상블 기댓값(%)', ascending=False).head(3).reset_index(drop=True)
                        df_cat.index += 1
                        st.dataframe(df_cat[['ETF명', '현재추세', '앙상블 기댓값(%)', '모델 상관계수']].style.format({
                            '앙상블 기댓값(%)': '{:.3f}%',
                            '모델 상관계수': '{:.3f}'
                        }), use_container_width=True)

        with tab3:
            st.subheader("🔥 캘린더 연동 타임머신 백테스트")
            if not df_daily.empty:
                valid_dates = sorted(df_daily['Date'].unique())
                min_date = datetime.strptime(valid_dates[0], '%Y-%m-%d').date()
                max_date = datetime.strptime(valid_dates[-1], '%Y-%m-%d').date()
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    target_date = st.date_input("🗓️ 백테스트 기준일 선택:", value=max_date, min_value=min_date, max_value=max_date)
                with col2:
                    period = st.radio("⏳ 시뮬레이션 기간 선택:", ["1일 (1영업일)", "1주 (5영업일)", "2주 (10영업일)", "1달 (20영업일)"], horizontal=True)
                
                target_date_str = target_date.strftime('%Y-%m-%d')
                days = 1 if "1일" in period else (5 if "1주" in period else (10 if "2주" in period else 20))
                
                df_filtered = df_daily[df_daily['Date'] <= target_date_str]
                
                if df_filtered.empty or len(df_filtered['Date'].unique()) == 0:
                    st.warning("선택하신 날짜에 해당하는 데이터가 부족합니다.")
                else:
                    unique_dates = sorted(df_filtered['Date'].unique())[-days:]
                    df_period = df_filtered[df_filtered['Date'].isin(unique_dates)]
                    
                    daily_model_returns, daily_market_returns, daily_max_returns, dates_list = [], [], [], []
                    target_date_portfolio = None 
                    
                    for d in unique_dates:
                        df_d = df_period[df_period['Date'] == d]
                        
                        past_portfolio = optimize_portfolio(df_d, target_col='Pred')
                        daily_port_return = float((past_portfolio['추천비중(%)'] / 100 * past_portfolio['실제수익률(%)'].fillna(0)).sum())
                        daily_model_returns.append(daily_port_return / 100)
                        
                        if d == unique_dates[-1]:
                            target_date_portfolio = past_portfolio 
                        
                        daily_market_returns.append(float(df_d['Actual'].mean() / 100))
                        
                        max_portfolio = optimize_portfolio(df_d, target_col='Actual')
                        daily_max_return = float((max_portfolio['추천비중(%)'] / 100 * max_portfolio['실제수익률(%)'].fillna(0)).sum())
                        daily_max_returns.append(daily_max_return / 100)
                        
                        dates_list.append(d)
                    
                    cum_model = (np.cumprod(1 + np.array(daily_model_returns)) - 1) * 100
                    cum_market = (np.cumprod(1 + np.array(daily_market_returns)) - 1) * 100
                    cum_max = (np.cumprod(1 + np.array(daily_max_returns)) - 1) * 100
                    
                    st.divider()
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric(f"🤖 모델 포트폴리오 성과", f"{cum_model[-1]:.2f}%")
                    c2.metric(f"📊 시장 평균 1/N 성과", f"{cum_market[-1]:.2f}%")
                    c3.metric("✨ 알파 (초과 수익)", f"{(cum_model[-1] - cum_market[-1]):.2f}%")
                    c4.metric("👑 신의 영역 (이론적 최대)", f"{cum_max[-1]:.2f}%")
                    
                    df_chart = pd.DataFrame({
                        '👑 신의 영역 (완벽한 예지력)': cum_max,
                        '🤖 동적 가중치 앙상블 모델': cum_model, 
                        '📊 시장 전체 평균 1/N': cum_market
                    }, index=dates_list)
                    st.line_chart(df_chart, use_container_width=True)

                    st.markdown(f"### 🔍 기준일({dates_list[-1]})의 포트폴리오 매매 상세 내역")
                    if target_date_portfolio is not None:
                        st.dataframe(target_date_portfolio.style.format({
                            '추천비중(%)': '{:.1f}%', 
                            '앙상블 기댓값(%)': '{:.3f}%',
                            '실제수익률(%)': '{:.3f}%'
                        }), use_container_width=True)

        with tab4:
            st.subheader("🔍 ETF 개별 종목 정밀 분석 및 통계 지표")
            target_etf = st.selectbox("종목을 선택하세요:", df_analysis['ETF명'].unique())
            row = df_analysis[df_analysis['ETF명'] == target_etf].iloc[0]
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("현재 추세", row['현재추세'])
            c2.metric("내일 앙상블 기댓값", f"{row['앙상블 기댓값(%)']:.3f}%")
            c3.metric("단일 종목 규정 상한선", "최대 20.0%")
            c4.metric(f"최근 가중치 윈도우 상관계수", f"{row['모델 상관계수']:.3f}")
            
            ticker_clean = str(row['종목코드']).replace('A', '')
            try:
                with st.spinner(f"📈 {target_etf} 차트 로드 중..."):
                    g_start = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
                    df_stock_chart = fdr.DataReader(ticker_clean, g_start)[['Close']].rename(columns={'Close': '마감 종가'})
                    df_stock_chart['20일 이동평균선'] = df_stock_chart['마감 종가'].rolling(20).mean()
                    st.line_chart(df_stock_chart, use_container_width=True)
            except Exception as e:
                st.error(f"⚠️ 차트 로드 실패: {e}")

        with tab5:
            st.subheader("📝 나의 실전 매매 기록장 (Excel 스타일)")
            df_user_log = pd.read_csv(log_filename)
            edited_user_df = st.data_editor(df_user_log, num_rows="dynamic", use_container_width=True)
            
            if st.button("💾 실전 매매 기록 저장하기", type="primary"):
                try:
                    edited_user_df.to_csv(log_filename, index=False, encoding='utf-8-sig')
                    st.success("🎉 기록이 성공적으로 보관되었습니다.")
                    st.rerun()
                except Exception as e:
                    st.error(f"⚠️ 저장 오류: {e}")
                    
            if len(edited_user_df) > 1:
                st.divider()
                try:
                    df_chart_user = edited_user_df.copy()
                    df_chart_user['나의 실제 계좌수익률(%)'] = pd.to_numeric(df_chart_user['나의 실제 계좌수익률(%)']).fillna(0.0)
                    df_chart_user = df_chart_user.sort_values(by='매매일자').set_index('매매일자')
                    st.line_chart(df_chart_user[['나의 실제 계좌수익률(%)']])
                except: pass
else:
    st.sidebar.error(f"❌ `{csv_filename}` 파일을 찾을 수 없습니다.")