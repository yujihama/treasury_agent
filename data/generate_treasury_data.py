import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import random

# 設定
start_date = date(2023, 7, 1)
end_date = date(2025, 6, 30)
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

# 通貨と国の設定
currencies = ['JPY', 'USD', 'EUR', 'GBP', 'AUD', 'SGD']
countries = {
    'JPY': ('Japan', 'JP'),
    'USD': ('USA', 'US'),
    'EUR': ('Germany', 'DE'),
    'GBP': ('UK', 'GB'),
    'AUD': ('Australia', 'AU'),
    'SGD': ('Singapore', 'SG')
}

# ダミー企業・銀行名
banks = {
    'JPY': 'Bank_A_Japan',
    'USD': 'Bank_B_USA',
    'EUR': 'Bank_C_Germany',
    'GBP': 'Bank_D_UK',
    'AUD': 'Bank_E_Australia',
    'SGD': 'Bank_F_Singapore'
}

customers = [
    'Company_Alpha_US', 'Company_Beta_DE', 'Company_Gamma_AU',
    'Company_Delta_UK', 'Company_Epsilon_SG', 'Company_Zeta_FR'
]

vendors = [
    'Supplier_A_US', 'Supplier_B_CN', 'Supplier_C_FR',
    'Supplier_D_DE', 'Supplier_E_SG', 'Supplier_F_UK'
]

# 1. 為替レート生成
def generate_exchange_rates():
    data = []
    
    # 基準レート（2023年7月1日時点）
    base_rates = {
        'USD/JPY': 145.0,
        'EUR/JPY': 160.0,
        'GBP/JPY': 185.0,
        'AUD/JPY': 98.0,
        'SGD/JPY': 108.0
    }
    
    for date_val in date_range:
        for pair, base_rate in base_rates.items():
            # ランダムウォークで為替レート生成
            days_from_start = (date_val.date() - start_date).days
            volatility = 0.01  # 1%の日次ボラティリティ
            rate = base_rate * (1 + np.random.normal(0, volatility) * np.sqrt(days_from_start / 365))
            
            data.append({
                'date': date_val.date(),
                'currency_pair': pair,
                'rate': round(rate, 2)
            })
    
    return pd.DataFrame(data)

# 2. 口座残高生成
def generate_account_balance():
    data = []
    account_configs = [
        ('ACJ001', 'Bank_A_Japan', 'JPY', 500000000, 'Japan', 'JP'),
        ('ACU001', 'Bank_B_USA', 'USD', 2000000, 'USA', 'US'),
        ('ACE001', 'Bank_C_Germany', 'EUR', 1500000, 'Germany', 'DE'),
        ('ACG001', 'Bank_D_UK', 'GBP', 800000, 'UK', 'GB'),
        ('ACA001', 'Bank_E_Australia', 'AUD', 1200000, 'Australia', 'AU'),
        ('ACS001', 'Bank_F_Singapore', 'SGD', 900000, 'Singapore', 'SG')
    ]
    
    for date_val in date_range:
        for account_id, bank_name, currency, base_balance, country_name, country_code in account_configs:
            # 残高の変動（±10%程度）
            variation = np.random.normal(1.0, 0.1)
            balance = int(base_balance * variation)
            
            data.append({
                'date': date_val.date(),
                'account_id': account_id,
                'bank_name': bank_name,
                'currency': currency,
                'balance': balance,
                'country_name': country_name,
                'country_code': country_code
            })
    
    return pd.DataFrame(data)

# 3. 取引履歴生成
def generate_transactions():
    data = []
    transaction_types = ['payment', 'receipt', 'fee', 'transfer']
    
    # 2年間で約1500件の取引を生成
    for i in range(1500):
        transaction_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
        account_id = random.choice(['ACJ001', 'ACU001', 'ACE001', 'ACG001', 'ACA001', 'ACS001'])
        
        # 口座IDに応じて通貨を決定
        currency_map = {
            'ACJ001': 'JPY', 'ACU001': 'USD', 'ACE001': 'EUR',
            'ACG001': 'GBP', 'ACA001': 'AUD', 'ACS001': 'SGD'
        }
        currency = currency_map[account_id]
        
        # 金額の範囲を通貨に応じて調整
        if currency == 'JPY':
            amount = random.randint(100000, 50000000)
        else:
            amount = random.randint(1000, 500000)
        
        data.append({
            'transaction_date': transaction_date,
            'account_id': account_id,
            'transaction_id': f'TXN-{i+1:06d}',
            'transaction_type': random.choice(transaction_types),
            'amount': amount,
            'currency': currency,
            'description': f'Transaction_{i+1}',
            'counterparty_country_name': random.choice(list(countries.values()))[0],
            'counterparty_country_code': random.choice(list(countries.values()))[1]
        })
    
    return pd.DataFrame(data)

# 4. 売掛金生成
def generate_accounts_receivable():
    data = []
    
    for i in range(400):
        issue_date = start_date + timedelta(days=random.randint(0, 600))
        due_date = issue_date + timedelta(days=random.randint(30, 90))
        currency = random.choice(['USD', 'EUR', 'GBP', 'AUD', 'SGD'])
        
        if currency == 'JPY':
            amount = random.randint(1000000, 100000000)
        else:
            amount = random.randint(10000, 1000000)
        
        status = 'outstanding' if due_date > date.today() else random.choice(['outstanding', 'paid'])
        
        data.append({
            'invoice_id': f'INV-{currency}-{i+1:03d}',
            'customer_id': f'CUST-{i+1:03d}',
            'customer_name': random.choice(customers),
            'issue_date': issue_date,
            'due_date': due_date,
            'amount': amount,
            'currency': currency,
            'status': status
        })
    
    return pd.DataFrame(data)

# 5. 買掛金生成
def generate_accounts_payable():
    data = []
    
    for i in range(600):
        received_date = start_date + timedelta(days=random.randint(0, 600))
        due_date = received_date + timedelta(days=random.randint(30, 90))
        currency = random.choice(['USD', 'EUR', 'GBP', 'AUD', 'SGD'])
        
        if currency == 'JPY':
            amount = random.randint(500000, 50000000)
        else:
            amount = random.randint(5000, 500000)
        
        status = 'unpaid' if due_date > date.today() else random.choice(['unpaid', 'paid'])
        
        data.append({
            'invoice_id': f'AP-{currency}-{i+1:03d}',
            'vendor_id': f'VEND-{i+1:03d}',
            'vendor_name': random.choice(vendors),
            'received_date': received_date,
            'due_date': due_date,
            'amount': amount,
            'currency': currency,
            'status': status
        })
    
    return pd.DataFrame(data)

# 6. 投資・有価証券生成
def generate_investments():
    data = [
        {
            'security_id': 'SEC-US-001',
            'security_name': 'US_Treasury_10Y',
            'asset_class': 'bond',
            'acquisition_date': date(2023, 8, 1),
            'quantity': 1000,
            'acquisition_price': 100.0,
            'market_price': 98.5,
            'currency': 'USD'
        },
        {
            'security_id': 'SEC-DE-001',
            'security_name': 'German_Bund_10Y',
            'asset_class': 'bond',
            'acquisition_date': date(2023, 9, 1),
            'quantity': 500,
            'acquisition_price': 102.0,
            'market_price': 101.2,
            'currency': 'EUR'
        },
        {
            'security_id': 'SEC-GB-001',
            'security_name': 'UK_Corporate_Stock',
            'asset_class': 'stock',
            'acquisition_date': date(2024, 1, 15),
            'quantity': 2000,
            'acquisition_price': 42.50,
            'market_price': 45.20,
            'currency': 'GBP'
        },
        {
            'security_id': 'SEC-AU-001',
            'security_name': 'Australian_Mining_ETF',
            'asset_class': 'etf',
            'acquisition_date': date(2024, 3, 1),
            'quantity': 1500,
            'acquisition_price': 85.00,
            'market_price': 88.75,
            'currency': 'AUD'
        },
        {
            'security_id': 'SEC-SG-001',
            'security_name': 'Singapore_REIT',
            'asset_class': 'reit',
            'acquisition_date': date(2024, 5, 1),
            'quantity': 3000,
            'acquisition_price': 1.20,
            'market_price': 1.35,
            'currency': 'SGD'
        }
    ]
    
    return pd.DataFrame(data)

# 7. 借入金生成
def generate_loans():
    data = [
        {
            'loan_id': 'LN-US-01',
            'lender_name': 'Mega_Bank_USA',
            'loan_date': date(2023, 10, 1),
            'maturity_date': date(2030, 12, 31),
            'principal_amount': 10000000,
            'outstanding_balance': 8500000,
            'currency': 'USD',
            'interest_rate': 4.5,
            'interest_type': 'fixed'
        },
        {
            'loan_id': 'LN-EU-01',
            'lender_name': 'European_Central_Bank',
            'loan_date': date(2024, 2, 1),
            'maturity_date': date(2028, 6, 30),
            'principal_amount': 5000000,
            'outstanding_balance': 4200000,
            'currency': 'EUR',
            'interest_rate': 3.8,
            'interest_type': 'variable'
        },
        {
            'loan_id': 'LN-GB-01',
            'lender_name': 'London_Finance_Corp',
            'loan_date': date(2023, 12, 15),
            'maturity_date': date(2026, 12, 15),
            'principal_amount': 2000000,
            'outstanding_balance': 1600000,
            'currency': 'GBP',
            'interest_rate': 5.2,
            'interest_type': 'fixed'
        },
        {
            'loan_id': 'LN-AU-01',
            'lender_name': 'Australia_National_Bank',
            'loan_date': date(2024, 4, 1),
            'maturity_date': date(2027, 4, 1),
            'principal_amount': 3000000,
            'outstanding_balance': 2800000,
            'currency': 'AUD',
            'interest_rate': 4.8,
            'interest_type': 'variable'
        }
    ]
    
    return pd.DataFrame(data)

# 8. デリバティブ取引生成
def generate_derivatives():
    data = []
    
    # 為替予約の生成
    currency_pairs = ['USD/JPY', 'EUR/JPY', 'GBP/JPY', 'AUD/JPY', 'SGD/JPY']
    positions = ['buy_foreign', 'sell_foreign']
    
    for i in range(60):
        trade_date = start_date + timedelta(days=random.randint(0, 500))
        maturity_date = trade_date + timedelta(days=random.randint(30, 180))
        currency_pair = random.choice(currency_pairs)
        position = random.choice(positions)
        
        # 契約レートの設定（基準レート±5%程度）
        base_rates = {
            'USD/JPY': 145.0,
            'EUR/JPY': 160.0,
            'GBP/JPY': 185.0,
            'AUD/JPY': 98.0,
            'SGD/JPY': 108.0
        }
        
        contract_rate = base_rates[currency_pair] * (1 + np.random.normal(0, 0.05))
        notional_principal = random.randint(100000, 2000000)
        
        data.append({
            'contract_id': f'FWD-{currency_pair.replace("/", "")}-{i+1:03d}',
            'trade_date': trade_date,
            'maturity_date': maturity_date,
            'derivative_type': 'FX_Forward',
            'notional_principal': notional_principal,
            'currency_pair': currency_pair,
            'contract_rate': round(contract_rate, 2),
            'position': position
        })
    
    return pd.DataFrame(data)

# データ生成と保存
print("為替レートデータを生成中...")
exchange_rates_df = generate_exchange_rates()
exchange_rates_df.to_csv('/home/ubuntu/exchange_rates.csv', index=False)
print(f"為替レートデータ: {len(exchange_rates_df)}行")

print("口座残高データを生成中...")
account_balance_df = generate_account_balance()
account_balance_df.to_csv('/home/ubuntu/account_balance.csv', index=False)
print(f"口座残高データ: {len(account_balance_df)}行")

print("取引履歴データを生成中...")
transactions_df = generate_transactions()
transactions_df.to_csv('/home/ubuntu/transactions.csv', index=False)
print(f"取引履歴データ: {len(transactions_df)}行")

print("売掛金データを生成中...")
accounts_receivable_df = generate_accounts_receivable()
accounts_receivable_df.to_csv('/home/ubuntu/accounts_receivable.csv', index=False)
print(f"売掛金データ: {len(accounts_receivable_df)}行")

print("買掛金データを生成中...")
accounts_payable_df = generate_accounts_payable()
accounts_payable_df.to_csv('/home/ubuntu/accounts_payable.csv', index=False)
print(f"買掛金データ: {len(accounts_payable_df)}行")

print("投資データを生成中...")
investments_df = generate_investments()
investments_df.to_csv('/home/ubuntu/investments.csv', index=False)
print(f"投資データ: {len(investments_df)}行")

print("借入金データを生成中...")
loans_df = generate_loans()
loans_df.to_csv('/home/ubuntu/loans.csv', index=False)
print(f"借入金データ: {len(loans_df)}行")

print("デリバティブデータを生成中...")
derivatives_df = generate_derivatives()
derivatives_df.to_csv('/home/ubuntu/derivatives.csv', index=False)
print(f"デリバティブデータ: {len(derivatives_df)}行")

print("\nすべてのCSVファイルが正常に生成されました！")

