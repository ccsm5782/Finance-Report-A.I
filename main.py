# daily_analyzer.py
# -*- coding: utf-8 -*-
"""
일일 한국 주식시장 자동 분석 시스템
- pykrx 중심의 안정적 데이터 수집
- 일일 등락률 기반 분석
- 기술적 분석 추가 (이동평균선, 지지저항, 거래량, 외국인/기관 매매)
"""
import time
from datetime import datetime, timedelta
import pandas as pd
import requests
from pykrx import stock
import warnings
import os
from dotenv import load_dotenv

# .env 파일에 정의된 환경 변수를 로드합니다.
load_dotenv()

warnings.filterwarnings('ignore')

# GenAI 초기화
_GENAI_MODE = None
_genai_client = None

def initialize_genai():
    """Google Gemini AI 모델을 초기화합니다."""
    global _GENAI_MODE, _genai_client
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        _genai_client = genai.GenerativeModel('gemini-2.5-flash')
        _GENAI_MODE = "google_genai"
        print("🤖 GenAI 초기화: google.genai 모드")
        return True
    except Exception as e:
        print(f"⚠️ google.genai 실패: {e}")
        _GENAI_MODE = None
        print("❌ GenAI 초기화 실패 - 템플릿 보고서 사용")
        return False


class DailyMarketAnalyzer:
    def __init__(self):
        """분석기 클래스를 초기화하고 주요 변수를 설정합니다."""
        self.naver_client_id = os.getenv("NAVER_CLIENT_ID")
        self.naver_client_secret = os.getenv("NAVER_CLIENT_SECRET")
        self.genai_available = initialize_genai()
        self.today, self.yesterday = self.get_analysis_dates()
        
        # 분석 결과 저장을 위한 변수
        self.valid_indices = []
        self.all_news = []
        self.performance_stats = {}
        self.technical_analysis_results = []

    def get_analysis_dates(self):
        """분석 기준이 되는 오늘과 어제 날짜를 영업일 기준으로 계산합니다."""
        today = datetime.now()
        
        # 주말 및 장 마감 시간(15시) 보정
        if today.weekday() == 5: today -= timedelta(days=1)
        elif today.weekday() == 6: today -= timedelta(days=2)
        
        yesterday = today - timedelta(days=1)
        while yesterday.weekday() > 4:
            yesterday -= timedelta(days=1)
        
        if datetime.now().hour < 15:
            today = yesterday
            yesterday = today - timedelta(days=1)
            while yesterday.weekday() > 4:
                yesterday -= timedelta(days=1)
        
        today_str = today.strftime("%Y%m%d")
        yesterday_str = yesterday.strftime("%Y%m%d")
        
        print(f"📊 일일 분석: {yesterday.strftime('%Y-%m-%d')} → {today.strftime('%Y-%m-%d')}")
        return today_str, yesterday_str

    # ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
    # 1단계: pykrx 지수 목록 수집
    # ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
    
    def fetch_market_indices(self):
        """KOSPI, KOSDAQ, KRX 시장의 모든 지수 티커와 이름을 수집합니다."""
        print("\n🔍 [1단계] pykrx 지수 목록 수집...")
        self.valid_indices = []
        
        for market in ["KOSPI", "KOSDAQ", "KRX"]:
            try:
                print(f"   {market} 지수 수집 중...")
                tickers = []
                try:
                    tickers = stock.get_index_ticker_list(self.today, market=market)
                except KeyError as e:
                    print(f"   ⚠️ '{e}' 오류 발생. 최신 데이터로 재시도합니다.")
                    tickers = stock.get_index_ticker_list(market=market)

                for ticker in tickers:
                    try:
                        name = stock.get_index_ticker_name(ticker)
                        if name and name.strip():
                            self.valid_indices.append({'pykrx_code': ticker, 'name': name.strip(), 'market': market})
                        time.sleep(0.01)
                    except Exception:
                        continue
                print(f"   ✅ {market}: {len(tickers)}개")
            except Exception as e:
                print(f"   ❌ {market} 실패: {e}")
        
        total_indices = len(self.valid_indices)
        print(f"📊 총 조회 가능 지수: {total_indices}개")
        return total_indices > 0

    # ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
    # 2단계: 일일 등락률 계산
    # ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
    
    def _calculate_index_return(self, pykrx_code, max_retries=3):
        """안전하게 지수의 일일 등락률을 계산하며, 실패 시 재시도합니다."""
        for attempt in range(max_retries):
            try:
                end_date = datetime.strptime(self.today, "%Y%m%d")
                start_date = end_date - timedelta(days=7) # 영업일 확보를 위해 7일 전부터 조회
                df = stock.get_index_ohlcv_by_date(start_date.strftime("%Y%m%d"), self.today, pykrx_code)
                
                if df is None or len(df) < 2:
                    start_date = end_date - timedelta(days=30)
                    df = stock.get_index_ohlcv_by_date(start_date.strftime("%Y%m%d"), self.today, pykrx_code)
                    if df is None or len(df) < 2: return None
                
                start_price = float(df.iloc[-2]["종가"])
                end_price = float(df.iloc[-1]["종가"])
                
                if start_price <= 0: return None
                return (end_price / start_price - 1.0) * 100.0
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"      ❌ {pykrx_code} 데이터 수집 실패: {str(e)[:50]}")
                    return None
                time.sleep(0.1 * (attempt + 1))
        return None

    def calculate_all_index_returns(self):
        """수집된 모든 지수의 일일 등락률을 계산하고 통계를 생성합니다."""
        print("\n📊 [2단계] 일일 등락률 계산...")
        results = []
        total = len(self.valid_indices)
        
        print(f"   총 {total}개 지수 처리 중...")
        for i, index_info in enumerate(self.valid_indices, 1):
            if i % max(1, total // 20) == 0 or i == total:
                print(f"   진행률: {i}/{total} ({i/total*100:.1f}%)")
            
            return_rate = self._calculate_index_return(index_info['pykrx_code'])
            if return_rate is not None:
                index_info['return_rate'] = return_rate
                results.append(index_info)
            time.sleep(0.01)
        
        if not results:
            print("❌ 모든 지수 데이터 수집 실패. 휴장일이거나 네트워크 문제일 수 있습니다.")
            return pd.DataFrame()
        
        df = pd.DataFrame(results).sort_values('return_rate', ascending=False)
        self.performance_stats = {
            'total_analyzed': len(df),
            'positive_count': len(df[df['return_rate'] > 0]),
            'negative_count': len(df[df['return_rate'] < 0]),
            'avg_return': df['return_rate'].mean(),
            'max_return': df['return_rate'].max(),
            'min_return': df['return_rate'].min()
        }
        
        print(f"✅ 일일 등락률 계산 완료: {len(results)}/{total}개 성공")
        return df
    
    # ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
    # 3단계: 구성종목 분석
    # ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
        
    def fetch_index_constituents(self, pykrx_code):
        """지수 코드를 받아 해당 지수의 구성 종목 목록을 DataFrame으로 반환합니다."""
        print(f"   🔍 구성종목 조회 (pykrx: {pykrx_code})")
        try:
            tickers = stock.get_index_portfolio_deposit_file(self.today, pykrx_code)
            
            is_empty = not tickers or (isinstance(tickers, pd.DataFrame) and tickers.empty)
            if is_empty:
                tickers = stock.get_index_portfolio_deposit_file(pykrx_code)

            is_empty_after_retry = not tickers or (isinstance(tickers, pd.DataFrame) and tickers.empty)
            if is_empty_after_retry:
                return pd.DataFrame()
            
            time.sleep(0.1)
            
            result = []
            for ticker in tickers:
                try:
                    name = stock.get_market_ticker_name(ticker)
                    if name: result.append({'ticker': ticker, 'name': f"{name} ({ticker})"})
                    time.sleep(0.05) 
                except Exception: continue
            
            print(f"   ✅ 구성종목: {len(result)}개")
            return pd.DataFrame(result)
        except Exception as e:
            print(f"   ❌ 구성종목 조회 중 예외 발생: {e}")
            return pd.DataFrame()

    def _calculate_stock_return(self, ticker, max_retries=2):
        """안전하게 개별 종목의 일일 등락률을 계산하며, 실패 시 재시도합니다."""
        for attempt in range(max_retries):
            try:
                end_date = datetime.strptime(self.today, "%Y%m%d")
                start_date = end_date - timedelta(days=7)
                df = stock.get_market_ohlcv_by_date(start_date.strftime("%Y%m%d"), self.today, ticker)
                
                if df is None or len(df) < 2:
                    start_date = end_date - timedelta(days=30)
                    df = stock.get_market_ohlcv_by_date(start_date.strftime("%Y%m%d"), self.today, ticker)
                    if df is None or len(df) < 2: return None
                
                start_price = float(df.iloc[-2]["종가"])
                end_price = float(df.iloc[-1]["종가"])
                
                if start_price <= 0: return None
                return (end_price / start_price - 1.0) * 100.0
            except Exception:
                if attempt == max_retries - 1: return None
                time.sleep(0.1)
        return None

    def analyze_top_bottom_indices(self, daily_returns):
        """등락률 상위 2개, 하위 2개 지수를 선정하고 심층 분석을 수행합니다."""
        print("\n🏆 상위/하위 지수 다중 분석...")
        if len(daily_returns) < 4:
            print("❌ 분석할 지수가 부족합니다 (최소 4개 필요)")
            return None
            
        top_2_indices = daily_returns.head(2).to_dict('records')
        bottom_2_indices = daily_returns.tail(2).to_dict('records')
        
        analysis_results = {'top_indices': [], 'bottom_indices': [], 'all_best_stocks': [], 'all_worst_stocks': []}
        
        print(f"📈 상위 2개 지수 분석:")
        for i, index in enumerate(top_2_indices, 1):
            print(f"   {i}. {index['name']} ({index['pykrx_code']}): {index['return_rate']:.2f}%")
            result = self._process_single_index_analysis(index, f"상위{i}위")
            if result:
                analysis_results['top_indices'].append(result)
                analysis_results['all_best_stocks'].extend(result['best_stocks'])
                analysis_results['all_worst_stocks'].extend(result['worst_stocks'])
        
        print(f"\n📉 하위 2개 지수 분석:")
        for i, index in enumerate(bottom_2_indices, 1):
            print(f"   {i}. {index['name']} ({index['pykrx_code']}): {index['return_rate']:.2f}%")
            result = self._process_single_index_analysis(index, f"하위{i}위")
            if result:
                analysis_results['bottom_indices'].append(result)
                analysis_results['all_best_stocks'].extend(result['best_stocks'])
                analysis_results['all_worst_stocks'].extend(result['worst_stocks'])
        
        return analysis_results

    def _process_single_index_analysis(self, index_info, rank_description):
        """단일 지수에 대해 구성 종목 성과 분석을 수행합니다."""
        print(f"     🔍 {rank_description} 지수 구성종목 분석...")
        constituents = self.fetch_index_constituents(index_info['pykrx_code'])
        if constituents.empty:
            print(f"     ❌ {rank_description} 구성종목 조회 실패")
            return None
        
        stock_performance = self.calculate_stock_performance(constituents, index_info['return_rate'])
        if stock_performance.empty:
            print(f"     ❌ {rank_description} 개별 종목 분석 실패")
            return None
        
        print(f"     ✅ {rank_description} 분석 완료: {len(stock_performance)}개 종목")
        
        return {
            'index_info': index_info,
            'rank': rank_description,
            'constituents_count': len(stock_performance),
            'outperformed_count': len(stock_performance[stock_performance['outperformed']]),
            'best_stocks': stock_performance.nlargest(3, 'relative_performance').to_dict('records'),
            'worst_stocks': stock_performance.nsmallest(3, 'relative_performance').to_dict('records')
        }

    # ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
    # 3.5단계: 기술적 분석
    # ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
        
    def fetch_technical_data(self, ticker, max_retries=2):
        """개별 종목의 기술적 분석을 위한 OHLCV 및 이동평균선 데이터를 수집합니다."""
        for attempt in range(max_retries):
            try:
                end_date = datetime.strptime(self.today, "%Y%m%d")
                start_date = end_date - timedelta(days=180)
                df = stock.get_market_ohlcv_by_date(start_date.strftime("%Y%m%d"), self.today, ticker)
                if df is None or len(df) < 120: return None
                
                current_price = float(df.iloc[-1]['종가'])
                current_volume = float(df.iloc[-1]['거래량'])
                
                return {
                    'current_price': current_price,
                    'ma_20': df['종가'].rolling(window=20).mean().iloc[-1],
                    'ma_60': df['종가'].rolling(window=60).mean().iloc[-1],
                    'ma_120': df['종가'].rolling(window=120).mean().iloc[-1],
                    'high_120': float(df['고가'].tail(120).max()),
                    'low_20': float(df['저가'].tail(20).min()),
                    'current_volume': current_volume,
                    'yesterday_volume': float(df.iloc[-2]['거래량']) if len(df) > 1 else current_volume,
                    'avg_volume_20': float(df['거래량'].tail(20).mean())
                }
            except Exception:
                if attempt == max_retries - 1: return None
                time.sleep(0.1)
        return None

    def fetch_institutional_data(self, ticker, max_retries=2):
        """개별 종목의 최근 5일간 외국인/기관 순매수 데이터를 수집합니다."""
        for attempt in range(max_retries):
            try:
                start_date = datetime.strptime(self.today, "%Y%m%d") - timedelta(days=10)
                df = stock.get_market_net_purchases_of_equities_by_investor(start_date.strftime("%Y%m%d"), self.today, ticker)

                return {
                    'foreigner_net_5d': float(df['외국인'].sum()) / 100000000,  # 억원 단위
                    'institution_net_5d': float(df['기관계'].sum()) / 100000000  # 억원 단위
                } if not df.empty else {'foreigner_net_5d': 0, 'institution_net_5d': 0}
            except Exception:
                if attempt == max_retries - 1: return None
                time.sleep(0.1)
        return None

    def analyze_stock_technicals(self, ticker, stock_name):
        """수집된 데이터를 바탕으로 개별 종목의 기술적 지표를 종합 분석합니다."""
        clean_ticker = ticker.split('(')[1].split(')')[0] if '(' in ticker and ')' in ticker else ticker
        tech_data = self.fetch_technical_data(clean_ticker)
        if not tech_data: return None
    
        inst_data = self.fetch_institutional_data(clean_ticker)
        if not inst_data: inst_data = {'foreigner_net_5d': 0, 'institution_net_5d': 0}
        
        current_price = tech_data['current_price']
        
        # 이동평균선 분석
        ma_analysis = []
        ma_20_pct = (current_price / tech_data['ma_20'] - 1) * 100
        ma_analysis.append(f"20일선 {'상향돌파' if ma_20_pct > 0 else '하회'}({ma_20_pct:+.1f}%)")
        ma_60_pct = (current_price / tech_data['ma_60'] - 1) * 100
        ma_analysis.append(f"60일선 {'상위' if ma_60_pct > 0 else '하회'}({ma_60_pct:+.1f}%)")
        
        arrangement = "혼재"
        if tech_data['ma_20'] > tech_data['ma_60'] > tech_data['ma_120']: arrangement = "정배열"
        elif tech_data['ma_20'] < tech_data['ma_60'] < tech_data['ma_120']: arrangement = "역배열"
        ma_analysis.append(arrangement)
        
        # 지지/저항, 거래량, 수급 분석
        recent_high_pct = (current_price / tech_data['high_120'] - 1) * 100
        low_20_pct = (current_price / tech_data['low_20'] - 1) * 100
        support_resistance = [f"120일 고점 대비 {recent_high_pct:+.1f}%", f"20일 저점 대비 {low_20_pct:+.1f}%"]
        
        volume_analysis = []
        if tech_data['yesterday_volume'] > 0:
            change_pct = (tech_data['current_volume'] / tech_data['yesterday_volume'] - 1) * 100
            status = "급증" if change_pct > 50 else ("증가" if change_pct > 0 else "감소")
            volume_analysis.append(f"전일대비 {abs(change_pct):.0f}% {status}")
        if tech_data['avg_volume_20'] > 0:
            avg_pct = (tech_data['current_volume'] / tech_data['avg_volume_20'] - 1) * 100
            volume_analysis.append(f"20일 평균대비 {avg_pct:+.0f}%")
        
        institutional = []
        if abs(inst_data['foreigner_net_5d']) >= 10: # 외국인 순매매 기준: 10억
            direction = "순매수" if inst_data['foreigner_net_5d'] > 0 else "순매도"
            institutional.append(f"외국인 5일 {direction} {abs(inst_data['foreigner_net_5d']):.0f}억")
        if abs(inst_data['institution_net_5d']) >= 5: # 기관 순매매 기준: 5억
            direction = "순매수" if inst_data['institution_net_5d'] > 0 else "순매도"
            institutional.append(f"기관 5일 {direction} {abs(inst_data['institution_net_5d']):.0f}억")
        if not institutional: institutional.append("외국인/기관 매매 미미")
        
        return {
            'ticker': clean_ticker, 'name': stock_name,
            'moving_average': ", ".join(ma_analysis),
            'support_resistance': ", ".join(support_resistance),
            'volume': ", ".join(volume_analysis) or "N/A",
            'institutional': ", ".join(institutional)
        }

    def run_technical_analysis(self, analysis_results):
        """선별된 주요 종목들에 대한 기술적 분석을 일괄 수행합니다."""
        print("\n📈 [3.5단계] 24개 종목 기술적 분석...")
        all_stocks = analysis_results.get('all_best_stocks', []) + analysis_results.get('all_worst_stocks', [])
        
        print(f"   총 {len(all_stocks)}개 종목 기술적 분석 수행...")
        for i, stock_info in enumerate(all_stocks, 1):
            if i % 6 == 0 or i == len(all_stocks):
                print(f"   진행률: {i}/{len(all_stocks)} ({i/len(all_stocks)*100:.1f}%)")
            
            tech_result = self.analyze_stock_technicals(stock_info['ticker'], stock_info['name'])
            if tech_result:
                self.technical_analysis_results.append(tech_result)
            time.sleep(0.02)
        
        print(f"✅ 기술적 분석 완료: {len(self.technical_analysis_results)}/{len(all_stocks)}개 성공")

    # ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
    # 4단계: 뉴스 수집
    # ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
    
    def _fetch_news_from_api(self, query, k=3):
        """네이버 뉴스 API를 사용해 뉴스를 검색하고 최신성, 중복을 필터링합니다."""
        try:
            headers = {"X-Naver-Client-Id": self.naver_client_id, "X-Naver-Client-Secret": self.naver_client_secret}
            params = {"query": query, "display": 30, "start": 1, "sort": "sim"}
            res = requests.get("https://openapi.naver.com/v1/search/news.json", headers=headers, params=params, timeout=10)
            res.raise_for_status()
            
            news, seen_titles = [], set()
            cutoff = datetime.strptime(self.yesterday, "%Y%m%d")
            
            for item in res.json().get("items", []):
                try:
                    pub_date = datetime.strptime(item["pubDate"], "%a, %d %b %Y %H:%M:%S %z").replace(tzinfo=None)
                    if pub_date >= cutoff:
                        title = item["title"].replace("<b>", "").replace("</b>", "")
                        if title[:50] not in seen_titles:
                            seen_titles.add(title[:50])
                            news.append({
                                "title": title,
                                "description": item["description"].replace("<b>", "").replace("</b>", ""),
                                "link": item["link"], "pub_date": pub_date.strftime("%Y-%m-%d %H:%M")
                            })
                            if len(news) >= k: return news
                except Exception: continue
            return news
        except Exception: return []

    def fetch_comprehensive_stock_news(self, stock_name, k=4):
        """'종목명+주식', '종목명+주가' 두 키워드로 종목 뉴스를 포괄적으로 수집합니다."""
        all_news, seen_titles = [], set()
        
        for keyword in [f"{stock_name} 주식", f"{stock_name} 주가"]:
            news_list = self._fetch_news_from_api(keyword, k=k//2 + 1)
            for news in news_list:
                if news["title"][:50] not in seen_titles:
                    seen_titles.add(news["title"][:50])
                    all_news.append(news)
            time.sleep(0.1)
        return all_news[:k]

    def calculate_stock_performance(self, constituents, index_return):
        """지수 구성 종목들의 개별 등락률을 계산하고 지수와 비교 분석합니다."""
        print(f"       📊 개별 종목 분석 (총 {len(constituents)}개)")
        if constituents.empty: return pd.DataFrame()
        
        results = []
        for i, row in enumerate(constituents.to_dict('records'), 1):
            if i % max(1, len(constituents) // 5) == 0 or i == len(constituents):
                print(f"         진행률: {i}/{len(constituents)} ({i/len(constituents)*100:.1f}%)")
            
            stock_return = self._calculate_stock_return(row['ticker'])
            if stock_return is not None:
                row.update({
                    'stock_return': stock_return, 'index_return': float(index_return),
                    'relative_performance': stock_return - float(index_return),
                    'outperformed': stock_return > float(index_return)
                })
                results.append(row)
            time.sleep(0.01)
        
        df = pd.DataFrame(results)
        if not df.empty: df = df.sort_values('relative_performance', ascending=False)
        print(f"       ✅ 개별 종목 분석 완료: {len(results)}개 성공")
        return df

    def collect_news_for_analysis(self, analysis_results):
        """분석 결과를 바탕으로 주요 지수 및 종목 관련 뉴스를 수집합니다."""
        print("\n📰 [4단계] 다중 지수 뉴스 수집...")
        self.all_news = []
        
        # 지수 뉴스 수집
        for idx_type in ['top_indices', 'bottom_indices']:
            for idx_data in analysis_results.get(idx_type, []):
                clean_name = idx_data['index_info']['name'].split('(')[0].strip()
                print(f"   {'📈' if idx_type == 'top_indices' else '📉'} {clean_name} 지수 뉴스")
                news_list = self._fetch_news_from_api(f"{clean_name} 지수", k=2)
                for news in news_list:
                    news.update({'category': idx_type, 'target': clean_name})
                    self.all_news.append(news)
                time.sleep(0.1)
        
        # 종목 뉴스 수집
        for stock_type in ['all_best_stocks', 'all_worst_stocks']:
            print(f"   {'📈' if stock_type == 'all_best_stocks' else '📉'} 전체 {'상위' if stock_type == 'all_best_stocks' else '하위'} 성과 종목 뉴스:")
            seen_stocks = set()
            for stock_data in analysis_results.get(stock_type, [])[:6]: # 최대 6개
                stock_name = stock_data['name'].split('(')[0].strip()
                if stock_name not in seen_stocks:
                    print(f"      - 뉴스 검색: '{stock_name}'")
                    seen_stocks.add(stock_name)
                    news_list = self.fetch_comprehensive_stock_news(stock_name, k=3)
                    for n in news_list:
                        n.update({'category': stock_type, 'target': stock_name})
                        self.all_news.append(n)
                    time.sleep(0.1)

        print(f"✅ 총 {len(self.all_news)}개 뉴스 수집 완료")
        return self.all_news

    # ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
    # 5단계: AI 보고서 생성
    # ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

    def create_ai_report_prompt(self, analysis_results):
        """AI 보고서 생성을 위한 상세한 프롬프트를 동적으로 생성합니다."""
        prompt = f"""한국 주식시장 일일 종합 분석 보고서를 작성하세요.

[분석 정보]
- 분석 날짜: {self.today[:4]}-{self.today[4:6]}-{self.today[6:]}
- 전일 대비 분석
- 상위 2개 지수 + 하위 2개 지수 종합 분석

[시장 전체 통계]
- 분석된 지수: {self.performance_stats.get('total_analyzed', 0)}개
- 상승 지수: {self.performance_stats.get('positive_count', 0)}개
- 하락 지수: {self.performance_stats.get('negative_count', 0)}개
- 평균 등락률: {self.performance_stats.get('avg_return', 0):.2f}%

[상위 성과 지수 TOP 2]
"""
        for i, idx in enumerate(analysis_results.get('top_indices', []), 1):
            info = idx.get('index_info', {})
            prompt += f"\n{i}. {info.get('name')} ({info.get('pykrx_code')}): {info.get('return_rate', 0):.2f}%"

        prompt += "\n\n[하위 성과 지수 TOP 2]\n"
        for i, idx in enumerate(analysis_results.get('bottom_indices', []), 1):
            info = idx.get('index_info', {})
            prompt += f"\n{i}. {info.get('name')} ({info.get('pykrx_code')}): {info.get('return_rate', 0):.2f}%"

        prompt += "\n\n[주요 종목 뉴스]\n"
        for news in self.all_news:
            prompt += f"\n- [{news.get('target')}] {news.get('title')} ({news.get('pub_date')})"
        
        prompt += "\n\n[주요 종목 기술적 분석]\n"
        for tech in self.technical_analysis_results:
            prompt += f"\n- {tech.get('name')}: 이평선({tech.get('moving_average')}), 거래량({tech.get('volume')}), 수급({tech.get('institutional')})"

        prompt += """\n\n[보고서 작성 지침]
- 다음의 3가지 마크다운 제목과 구조를 반드시 사용하여 총 3개의 문단으로 이루어진 보고서를 생성하세요.
- 각 섹션의 내용은 아래 가이드라인에 맞춰 논리적으로, 그리고 전문적인 금융 분석가 톤으로 작성하세요.
- 모든 지수와 종목명 뒤에는 괄호 안에 코드를 표기하세요. (예: 삼성전자(005930))
- 제시된 분량과 조건을 엄수하여 작성하세요.
---
## 1. 시장 전반 흐름 요약(7-10 문장으로 작성)
- 전체 지수, 상승/하락 지수 비율, 평균 등락률을 바탕으로 오늘 시장의 전반적인 분위기를 요약합니다.
- 가장 두드러진 성과를 보인 상위 2개 지수와 가장 부진했던 하위 2개 지수를 명시하여 시장의 핵심 동향을 제시합니다.
- 시장 전체에 영향을 미친 거시적 뉴스(예: 금리, 환율, 주요 경제 지표 발표)가 있다면 자세하게 언급합니다.
- 모든 문장에서 줄바꿈을 제거한 줄글 형태로 출력합니다.

## 2. 상위 지수 및 모멘텀 종목 심층 분석(7-10 문장으로 작성)
- 가장 높은 상승률을 보인 상위 2개 지수의 급등 원인을 섹터의 특징과 관련 뉴스를 통합하여 분석합니다.
- 각 지수별로, 상승을 주도한 핵심 '모멘텀 종목'들을 선정하여 언급합니다.
- 각 모멘텀 종목의 주가 급등 배경을 관련 뉴스, 기술적 분석 결과(예: 정배열, 20일선 돌파, 거래량 급증, 외국인 순매수)와 연결하여 앞으로의 투자 전략에 대해 설득력 있게 설명합니다.
- 모든 문장에서 줄바꿈을 제거한 줄글 형태로 출력합니다.

## 3. 하위 지수 및 저성장 종목 심층 분석(7-10 문장으로 작성)
- 가장 큰 하락률을 보인 하위 2개 지수의 부진 원인을 섹터가 마주한 리스크와 관련 뉴스를 통합하여 분석합니다.
- 각 지수별로, 하락을 주도한 '저성장 종목'들을 선정하여 언급합니다.
- 각 저성장 종목의 주가 하락 배경을 관련 뉴스(예: 실적 부진, 경쟁 심화)나 기술적 분석 결과(예: 역배열, 이평선 하회, 거래량 감소)와 연결하여 투자자가 유의해야 할 점과 앞으로의 투자 전략을 명확히 제시합니다.
- 모든 문장에서 줄바꿈을 제거한 줄글 형태로 출력합니다.
---"""
        return prompt

    def generate_ai_report(self, analysis_results):
        """AI 보고서 생성을 요청하고, 실패 시 템플릿 보고서를 반환합니다."""
        print("\n🤖 [5단계] 다중 지수 AI 보고서 생성...")
        prompt = self.create_ai_report_prompt(analysis_results)
        
        if self.genai_available:
            try:
                response = _genai_client.generate_content(prompt)
                report = response.text or ""
                if report:
                    print("✅ 다중 지수 AI 보고서 생성 완료")
                    return prompt, report
            except Exception as e:
                print(f"⚠️ AI 생성 실패: {e}")
        
        print("📄 AI 보고서 생성 실패. 기본 메시지를 반환합니다.")
        return prompt, "AI 보고서 생성에 실패했습니다."
        
    # ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
    # 6단계: 결과 저장 및 요약
    # ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
    
    def save_analysis_results(self, daily_returns, analysis_results, prompt, ai_report):
        """분석 과정에서 생성된 모든 데이터를 파일로 저장합니다."""
        print("\n💾 [6단계] 다중 지수 결과 저장...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        all_stocks = analysis_results.get('all_best_stocks', []) + analysis_results.get('all_worst_stocks', [])
        
        try:
            daily_returns.to_csv(f"daily_returns_{timestamp}.csv", index=False, encoding="utf-8-sig")
            if all_stocks: pd.DataFrame(all_stocks).to_csv(f"stock_performance_{timestamp}.csv", index=False, encoding="utf-8-sig")
            if self.all_news: pd.DataFrame(self.all_news).to_csv(f"news_data_{timestamp}.csv", index=False, encoding="utf-8-sig")
            if self.technical_analysis_results: pd.DataFrame(self.technical_analysis_results).to_csv(f"technical_analysis_{timestamp}.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame([self.performance_stats]).to_csv(f"market_stats_{timestamp}.csv", index=False, encoding="utf-8-sig")
            with open(f"ai_prompt_{timestamp}.txt", "w", encoding="utf-8") as f: f.write(prompt)
            with open(f"final_report_{timestamp}.md", "w", encoding="utf-8") as f: f.write(ai_report)
            print(f"✅ 7개 종류의 분석 파일 저장 완료")
        except Exception as e:
            print(f"❌ 파일 저장 실패: {e}")

    def print_summary(self, daily_returns, ai_report):
        """콘솔에 최종 분석 결과를 요약하여 출력합니다."""
        print("\n" + "=" * 80)
        print("✅ 다중 지수 일일 시장 분석 시스템 완료!")
        print("=" * 80)
        
        if self.performance_stats:
            stats = self.performance_stats
            print(f"   📊 분석 지수: {stats['total_analyzed']}개 | 상승: {stats['positive_count']}개 | 하락: {stats['negative_count']}개")
            print(f"   📈 평균 등락률: {stats['avg_return']:.2f}% | 최고: {stats['max_return']:.2f}% | 최저: {stats['min_return']:.2f}%")
        
        print(f"\n📈 일일 등락률 TOP 5:")
        for i, (_, row) in enumerate(daily_returns.head(5).iterrows(), 1):
            print(f"   {i}. {row['name']} ({row['pykrx_code']}): {row['return_rate']:.2f}%")
        
        print("\n" + "=" * 80)
        print("🤖 일일 시장 다중 지수 AI 투자 분석 보고서")
        print("=" * 80)
        print(ai_report)
        print("=" * 80)

    # ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
    # 메인 실행 로직
    # ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
    
    def run(self):
        """일일 시장 분석의 모든 단계를 순차적으로 실행합니다."""
        print("🚀 다중 지수 일일 한국 주식시장 분석 시스템")
        print("=" * 80)

        if not self.fetch_market_indices():
            print("❌ 지수 목록 수집 실패")
            return
        
        daily_returns = self.calculate_all_index_returns()
        if daily_returns.empty:
            print("❌ 등락률 계산 실패 - 데이터가 없습니다")
            return
        
        analysis_results = self.analyze_top_bottom_indices(daily_returns)
        if not analysis_results:
            print("❌ 다중 지수 분석 실패")
            return
        
        self.run_technical_analysis(analysis_results)
        self.collect_news_for_analysis(analysis_results)
        prompt, ai_report = self.generate_ai_report(analysis_results)
        
        self.save_analysis_results(daily_returns, analysis_results, prompt, ai_report)
        self.print_summary(daily_returns, ai_report)
        
        print("\n🎉 다중 지수 일일 시장 분석이 성공적으로 완료되었습니다!")

def main():
    """메인 실행 함수"""
    try:
        print(f"📦 pykrx: ✅")
    except ImportError as e:
        print(f"❌ 라이브러리 설치 필요: {e.name}. 'pip install {e.name}'를 실행하세요.")
        return
    
    analyzer = DailyMarketAnalyzer()
    analyzer.run()

if __name__ == "__main__":
    main()