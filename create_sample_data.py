import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# サンプルデータ作成用のスクリプト

def create_sample_data():
    """テスト用のサンプルCSVファイルを作成"""
    
    # 基本設定
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 国情報
    countries = [
        {'国名': '日本', '国コード': 'JPN'},
        {'国名': 'アメリカ', '国コード': 'USA'},
        {'国名': 'ドイツ', '国コード': 'DEU'},
        {'国名': 'イギリス', '国コード': 'GBR'},
        {'国名': 'フランス', '国コード': 'FRA'},
        {'国名': 'シンガポール', '国コード': 'SGP'},
        {'国名': 'オーストラリア', '国コード': 'AUS'},
        {'国名': 'カナダ', '国コード': 'CAN'},
        {'国名': 'スイス', '国コード': 'CHE'},
        {'国名': '韓国', '国コード': 'KOR'}
    ]
    
    # 1. 口座残高データ
    balances_data = []
    account_ids = ['ACC001', 'ACC002', 'ACC003', 'ACC004', 'ACC005', 'ACC006', 'ACC007', 'ACC008']
    banks = ['三菱UFJ銀行', 'みずほ銀行', '三井住友銀行', 'JPMorgan Chase', 'Deutsche Bank', 'HSBC', 'BNP Paribas', 'UBS']
    currencies = ['JPY', 'USD', 'EUR', 'GBP', 'SGD', 'AUD', 'CAD', 'CHF', 'KRW']
    
    for i, account_id in enumerate(account_ids):
        country = countries[i % len(countries)]
        bank = banks[i % len(banks)]
        currency = currencies[i % len(currencies)]
        
        # 各口座の残高推移を生成
        base_balance = random.uniform(1000000, 50000000)  # 100万〜5000万
        for date in date_range[::7]:  # 週次データ
            balance = base_balance + random.uniform(-base_balance*0.2, base_balance*0.2)
            balances_data.append({
                '日付': date.strftime('%Y-%m-%d'),
                '口座ID': account_id,
                '銀行名': bank,
                '通貨': currency,
                '残高': round(balance, 2),
                '国名': country['国名'],
                '国コード': country['国コード']
            })
    
    df_balances = pd.DataFrame(balances_data)
    df_balances.to_csv('sample_balances.csv', index=False, encoding='utf-8')
    
    # 2. 取引履歴データ
    transactions_data = []
    transaction_types = ['入金', '出金', '振替', '利息', '手数料', '為替取引']
    
    for i in range(1000):  # 1000件の取引
        date = date_range[random.randint(0, len(date_range)-1)]
        account_id = random.choice(account_ids)
        transaction_type = random.choice(transaction_types)
        
        # 取引金額の設定
        if transaction_type == '手数料':
            amount = -random.uniform(1000, 10000)
        elif transaction_type == '利息':
            amount = random.uniform(10000, 100000)
        elif transaction_type == '出金':
            amount = -random.uniform(100000, 5000000)
        else:
            amount = random.uniform(-2000000, 5000000)
        
        # 相手国の設定
        counterpart_country = random.choice(countries)
        
        transactions_data.append({
            '日付': date.strftime('%Y-%m-%d'),
            '口座ID': account_id,
            '取引ID': f'TXN{i+1:04d}',
            '取引タイプ': transaction_type,
            '金額': round(amount, 2),
            '摘要': f'{transaction_type}_{i+1}',
            '相手国名': counterpart_country['国名'],
            '相手国コード': counterpart_country['国コード']
        })
    
    df_transactions = pd.DataFrame(transactions_data)
    df_transactions.to_csv('sample_transactions.csv', index=False, encoding='utf-8')
    
    # 3. 予算データ
    budgets_data = []
    departments = ['営業部', '製造部', 'IT部', '人事部', '経理部', 'マーケティング部']
    categories = ['人件費', '設備投資', '運営費', '研究開発費', '広告宣伝費', '旅費交通費']
    
    for month in range(1, 13):
        for dept in departments:
            for category in categories:
                budget_amount = random.uniform(1000000, 10000000)
                budgets_data.append({
                    '日付': f'2024-{month:02d}-01',
                    '部門': dept,
                    'カテゴリ': category,
                    '予算額': round(budget_amount, 2)
                })
    
    df_budgets = pd.DataFrame(budgets_data)
    df_budgets.to_csv('sample_budgets.csv', index=False, encoding='utf-8')
    
    # 4. 為替レートデータ
    fx_rates_data = []
    currency_pairs = ['USD/JPY', 'EUR/JPY', 'GBP/JPY', 'AUD/JPY', 'EUR/USD', 'GBP/USD']
    base_rates = {'USD/JPY': 150, 'EUR/JPY': 160, 'GBP/JPY': 180, 'AUD/JPY': 100, 'EUR/USD': 1.08, 'GBP/USD': 1.25}
    
    for date in date_range[::1]:  # 日次データ
        for pair in currency_pairs:
            base_rate = base_rates[pair]
            rate = base_rate + random.uniform(-base_rate*0.05, base_rate*0.05)
            fx_rates_data.append({
                '日付': date.strftime('%Y-%m-%d'),
                '通貨ペア': pair,
                'レート': round(rate, 4)
            })
    
    df_fx_rates = pd.DataFrame(fx_rates_data)
    df_fx_rates.to_csv('sample_fx_rates.csv', index=False, encoding='utf-8')
    
    print("サンプルデータファイルが作成されました:")
    print("- sample_balances.csv (口座残高データ)")
    print("- sample_transactions.csv (取引履歴データ)")
    print("- sample_budgets.csv (予算データ)")
    print("- sample_fx_rates.csv (為替レートデータ)")
    
    return df_balances, df_transactions, df_budgets, df_fx_rates

if __name__ == "__main__":
    create_sample_data()

