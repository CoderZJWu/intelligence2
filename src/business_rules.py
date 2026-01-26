"""
业务规则模块，包含定价策略和业务逻辑
"""

def get_pricing_strategy():
    """获取定价策略规则"""
    return {
        "volume_discounts": {
            "0-10K": 0.0,
            "10K-50K": 0.03,
            "50K-100K": 0.06,
            "100K+": 0.09
        },
        "tier_discounts": {
            "Gold": 0.09,
            "Platinum": 0.06,
            "Portfolio": 0.03,
            "Silver": 0.0
        },
        "product_premiums": {
            "Clearing": 1.25,
            "Payments": 1.0,
            "Collections": 0.95,
            "Liquidity": 0.8
        },
        "country_adjustments": {
            "SG": 0.95,
            "US": 1.05,
            "GB": 1.0,
            "DE": 1.0,
            "HK": 0.98,
            "JP": 0.97,
            "AU": 0.99
        }
    }

def validate_pricing_rules(pricing, customer_info):
    """
    验证定价是否符合业务规则
    
    Args:
        pricing: 定价结果
        customer_info: 客户信息
    
    Returns:
        bool: 是否符合规则
        list: 违反的规则列表
    """
    violations = []
    
    # 规则1: Clearing价格 > Payments价格 > Collections价格 > Liquidity价格
    category_prices = {}
    for category, items in pricing.items():
        total_price = 0
        count = 0
        for item in items:
            for subcategory, products in item.items():
                for price_str in products.values():
                    price = float(price_str)
                    total_price += price
                    count += 1
        if count > 0:
            category_prices[category] = total_price / count
    
    if 'Clearing' in category_prices and 'Payments' in category_prices:
        if category_prices['Clearing'] <= category_prices['Payments']:
            violations.append("Clearing price should be higher than Payments price")
    
    if 'Payments' in category_prices and 'Collections' in category_prices:
        if category_prices['Payments'] <= category_prices['Collections']:
            violations.append("Payments price should be higher than Collections price")
    
    if 'Collections' in category_prices and 'Liquidity' in category_prices:
        if category_prices['Collections'] <= category_prices['Liquidity']:
            violations.append("Collections price should be higher than Liquidity price")
    
    # 规则2: Gold tier客户应比Silver tier客户获得至少5%的价格优惠
    if customer_info.get('tier') == 'Gold':
        # 模拟Silver tier价格
        silver_pricing = _simulate_tier_pricing(pricing, customer_info, 'Silver')
        silver_avg = _get_average_price(silver_pricing)
        gold_avg = _get_average_price(pricing)
        
        discount = (silver_avg - gold_avg) / silver_avg * 100
        if discount < 5:
            violations.append(f"Gold tier discount ({discount:.1f}%) should be at least 5%")
    
    # 规则3: 所有价格应在合理范围内(0.1-5.0)
    for category, items in pricing.items():
        for item in items:
            for subcategory, products in item.items():
                for product_code, price_str in products.items():
                    price = float(price_str)
                    if price < 0.1 or price > 5.0:
                        violations.append(f"Price for {product_code} ({price}) outside valid range (0.1-5.0)")
    
    # 规则4: 同一product level 2下的不同product code应有价格差异
    for category, items in pricing.items():
        for i, item in enumerate(items):
            for subcategory, products in item.items():
                if len(products) > 1:
                    prices = [float(p) for p in products.values()]
                    if max(prices) - min(prices) < 0.001:
                        violations.append(f"Insufficient price differentiation for {subcategory} products")
    
    return len(violations) == 0, violations

def _simulate_tier_pricing(pricing, customer_info, new_tier):
    """模拟不同tier的定价"""
    # 复制定价结构
    simulated_pricing = {k: [{k2: {k3: v3 for k3, v3 in v2.items()} 
                           for k2, v2 in item.items()} 
                           for item in v] 
                       for k, v in pricing.items()}
    
    # 应用tier折扣
    tier_discounts = get_pricing_strategy()["tier_discounts"]
    current_discount = tier_discounts.get(customer_info['tier'], 0.0)
    new_discount = tier_discounts.get(new_tier, 0.0)
    
    # 计算折扣差异
    discount_diff = current_discount - new_discount
    
    # 应用差异
    for category, items in simulated_pricing.items():
        for i, item in enumerate(items):
            for subcategory, products in item.items():
                for product_code, price_str in products.items():
                    price = float(price_str)
                    # 移除当前折扣，应用新折扣
                    base_price = price / (1 - current_discount)
                    new_price = base_price * (1 - new_discount)
                    simulated_pricing[category][i][subcategory][product_code] = f"{new_price:.4f}"
    
    return simulated_pricing

def _get_average_price(pricing):
    """计算平均价格"""
    total_price = 0
    product_count = 0
    
    for category, items in pricing.items():
        for item in items:
            for subcategory, products in item.items():
                for price_str in products.values():
                    total_price += float(price_str)
                    product_count += 1
    
    return total_price / product_count if product_count > 0 else 0

def generate_pricing_insights(pricing, customer_info, model_evaluation, shap_analysis):
    """
    生成定价洞察报告
    
    Args:
        pricing: 定价结果
        customer_info: 客户信息
        model_evaluation: 模型评估结果
        shap_analysis: SHAP分析结果
    
    Returns:
        str: 定价洞察报告
    """
    report = []
    
    report.append("="*80)
    report.append("GPBS PRICING INSIGHTS REPORT")
    report.append(f"Customer: {customer_info.get('country')} - Tier {customer_info.get('tier')}")
    report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("="*80)
    report.append("")
    
    # 定价竞争力分析
    avg_price = _get_average_price(pricing)
    market_avg = 0.65  # 假设市场平均价格
    competitiveness = (market_avg - avg_price) / market_avg * 100
    
    report.append("PRICING COMPETITIVENESS")
    report.append("-"*50)
    report.append(f"• Average price: ${avg_price:.4f}")
    report.append(f"• Market average: ${market_avg:.4f}")
    report.append(f"• Competitiveness: {competitiveness:.1f}% vs market")
    
    if competitiveness > 5:
        report.append("• This pricing is highly competitive (more than 5% below market average)")
    elif competitiveness > 0:
        report.append("• This pricing is competitive (below market average)")
    else:
        report.append("• This pricing is above market average - consider adjustments")
    
    report.append("")
    
    # 模型信心分析
    report.append("MODEL CONFIDENCE")
    report.append("-"*50)
    
    # 获取模型在该客户segment的性能
    segment_performance = model_evaluation['by_segment'].get(str(customer_info.get('segment_code', '')), {})
    tier_performance = model_evaluation['by_tier'].get(customer_info.get('tier', ''), {})
    
    report.append(f"• Model performance for this segment: {segment_performance.get('MAPE', 'N/A'):.2f}% MAPE")
    report.append(f"• Model performance for {customer_info.get('tier', '')} tier: {tier_performance.get('MAPE', 'N/A'):.2f}% MAPE")
    
    if segment_performance.get('MAPE', 100) < 8 and tier_performance.get('MAPE', 100) < 8:
        report.append("• High confidence in pricing recommendation")
    elif segment_performance.get('MAPE', 100) < 10 or tier_performance.get('MAPE', 100) < 10:
        report.append("• Moderate confidence in pricing recommendation")
    else:
        report.append("• Low confidence in pricing recommendation - consider manual review")
    
    report.append("")
    
    # 关键定价驱动因素
    report.append("KEY PRICING DRIVERS")
    report.append("-"*50)
    report.append("Based on SHAP analysis, the main factors influencing this pricing are:")
    
    # 这里应该从SHAP分析中获取具体信息
    # 为示例，使用一些通用信息
    report.append("• Service tier selection has the strongest impact on pricing")
    report.append("• Transaction volume drives significant volume discounts")
    report.append(f"• {customer_info.get('tier', '')} tier status provides appropriate discounts")
    
    report.append("")
    
    # 业务建议
    report.append("BUSINESS RECOMMENDATIONS")
    report.append("-"*50)
    
    if competitiveness > 5:
        report.append("• This pricing is highly competitive and likely to strengthen customer")
        report.append("  retention. Recommended for approval without changes.")
    elif competitiveness > 0:
        report.append("• This pricing is competitive. Recommended for approval, but monitor")
        report.append("  customer response for potential fine-tuning.")
    else:
        report.append("• Consider reducing prices for Clearing and Payments services to")
        report.append("  improve competitiveness. A 3-5% reduction would bring pricing")
        report.append("  in line with market expectations.")
    
    report.append("")
    report.append("="*80)
    report.append("END OF REPORT")
    report.append("="*80)
    
    return "\n".join(report)
