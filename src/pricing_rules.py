"""
业务规则验证模块，用于确保预测价格符合银行业务逻辑。
"""

def validate_price_structure(predictions):
    """
    验证价格结构是否符合业务规则
    
    规则：
    1. Clearing价格 > Payments价格 > Collections价格 > Liquidity价格
    2. Gold tier客户应比Silver tier客户获得至少5%的价格优惠
    3. 所有价格应在合理范围内(0.1-5.0)
    4. 同一product level 2下的不同product code应有价格差异
    """
    # 规则1: 产品类别间价格层级
    category_prices = {}
    for category, items in predictions.items():
        total_price = 0
        count = 0
        for item in items:
            for subcategory, products in item.items():
                for product_code, price_str in products.items():
                    price = float(price_str)
                    total_price += price
                    count += 1
        if count > 0:
            category_prices[category] = total_price / count
    
    # 检查Clearing > Payments > Collections > Liquidity
    clearing_price = category_prices.get('Clearing', 0)
    payments_price = category_prices.get('Payments', 0)
    collections_price = category_prices.get('Collections', 0)
    liquidity_price = category_prices.get('Liquidity', 0)
    
    if not (clearing_price >= payments_price >= collections_price >= liquidity_price):
        # 应用温和的调整，保持相对比例
        max_price = max(clearing_price, payments_price, collections_price, liquidity_price)
        min_price = min(clearing_price, payments_price, collections_price, liquidity_price)
        
        if max_price > 0:
            # 保持相对比例，但确保层级正确
            if clearing_price < payments_price:
                clearing_price = payments_price * 1.05
            if payments_price < collections_price:
                payments_price = collections_price * 1.02
            if collections_price < liquidity_price:
                collections_price = liquidity_price * 1.01
            
            # 应用调整到原始预测
            for category, target_price in [('Clearing', clearing_price), 
                                         ('Payments', payments_price),
                                         ('Collections', collections_price),
                                         ('Liquidity', liquidity_price)]:
                if category in category_prices and category_prices[category] > 0:
                    ratio = target_price / category_prices[category]
                    for i, item in enumerate(predictions.get(category, [])):
                        for subcategory, products in item.items():
                            for product_code in products.keys():
                                original_price = float(predictions[category][i][subcategory][product_code])
                                new_price = original_price * ratio
                                # 限制在合理范围
                                new_price = max(0.1, min(5.0, new_price))
                                predictions[category][i][subcategory][product_code] = f"{new_price:.4f}"
    
    # 规则3: 价格范围
    for category, items in predictions.items():
        for i, item in enumerate(items):
            for subcategory, products in item.items():
                for product_code, price_str in products.items():
                    price = float(price_str)
                    if price < 0.1 or price > 5.0:
                        # 温和调整到边界
                        price = max(0.1, min(5.0, price))
                        predictions[category][i][subcategory][product_code] = f"{price:.4f}"
    
    # 规则4: 同一product level 2下的价格差异
    for category, items in predictions.items():
        for i, item in enumerate(items):
            for subcategory, products in item.items():
                if len(products) > 1:
                    prices = [float(p) for p in products.values()]
                    min_price = min(prices)
                    max_price = max(prices)
                    
                    # 如果价格差异太小，添加细微差异
                    if max_price - min_price < 0.001:
                        base_price = sum(prices) / len(prices)
                        for j, product_code in enumerate(products.keys()):
                            # 添加基于产品代码的细微差异
                            variation = 0.001 * (j - len(products)/2) / len(products)
                            new_price = base_price + variation
                            predictions[category][i][subcategory][product_code] = f"{new_price:.4f}"
    
    return predictions

def apply_tier_discount(predictions, tier, base_tier='Silver'):
    """
    根据客户tier应用价格折扣
    
    Gold: 8-10% discount
    Platinum: 5-7% discount
    Portfolio: 2-4% discount
    Silver: no discount (base)
    """
    if tier == 'Gold':
        discount = 0.09  # 9% discount
    elif tier == 'Platinum':
        discount = 0.06  # 6% discount
    elif tier == 'Portfolio':
        discount = 0.03  # 3% discount
    else:  # Silver
        discount = 0.0
    
    # 应用折扣
    for category, items in predictions.items():
        for i, item in enumerate(items):
            for subcategory, products in item.items():
                for product_code, price_str in products.items():
                    price = float(price_str)
                    discounted_price = price * (1 - discount)
                    predictions[category][i][subcategory][product_code] = f"{discounted_price:.4f}"
    
    return predictions

def get_tier_comparison(base_predictions, request_tier):
    """
    生成不同tier的价格比较
    """
    tiers = ['Gold', 'Platinum', 'Portfolio', 'Silver']
    comparisons = {}
    
    for tier in tiers:
        # 复制基础预测
        tier_predictions = {k: [{k2: {k3: v3 for k3, v3 in v2.items()} 
                               for k2, v2 in item.items()} 
                               for item in v] 
                           for k, v in base_predictions.items()}
        
        # 应用tier折扣
        tier_predictions = apply_tier_discount(tier_predictions, tier)
        
        # 计算与请求tier的差异百分比
        if tier != request_tier:
            diff_percent = {}
            for category in base_predictions:
                if category in tier_predictions:
                    base_avg = sum([float(p) for items in base_predictions[category] 
                                   for item in items.values() for p in item.values()]) / len(base_predictions[category])
                    tier_avg = sum([float(p) for items in tier_predictions[category] 
                                   for item in items.values() for p in item.values()]) / len(tier_predictions[category])
                    diff_percent[category] = (tier_avg - base_avg) / base_avg * 100
            
            comparisons[tier] = {
                "price_difference_percent": diff_percent,
                "example_product": next(iter(next(iter(base_predictions.values()))[0].values()))
            }
    
    return comparisons
