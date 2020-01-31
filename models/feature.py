# -*- coding: utf-8 -*-

import re

subject_pat = {
    'lender_replace': re.compile(r'原告(?!：)'),
    'borrower_replace': re.compile(r'被告(?!：)'),
    'legal_represent_replace': re.compile(r'法定代表人(?!：)'),
    'lender': re.compile(r'原告：(.+?)[。|，]'),
    'borrower': re.compile(r'被告：(.+?)[。|，]'),
    'legal_represent': re.compile(r'法定代表人：(.+?)[。|，]')
}

interest_pat = [
    (re.compile(r'(月利息|月息|月利率|月利息率)按?(\d+(\.\d{1,2})?)([％分])'), 12),
    (re.compile(r"(月利息|月息|月利率|月利息率)按?(\d+(\.\d{1,2})?)毛"), 120),
    (re.compile(r"(月)(\d+(\.\d{1,2})?)％(利息|息|利率|利息率)"), 12),
    (re.compile(r"(年利息|年息|年利率|年利息率)按?(\d+(\.\d{1,2})?)([％分])"), 1),
    (re.compile(r"(年利息|年息|年利率|年利息率)按?(\d+(\.\d{1,2})?)毛"), 10),
    (re.compile(r"(年)(\d+(\.\d{1,2})?)％(利息|息|利率|利息率)"), 1)
]

interest_interval = [0.36, 0.24, 0]

payment_list = [['微信转账', '微信支付', '支付宝'], ['银行转账', '手机银行'], ['现金']]

agreement_list = [['合同', '协议'], ['收据', '凭据', '借条', '书面承诺', '承诺书'], ['流水'], [], ['微信', '短信', '聊天']]


def extract_features_and_replace(text):
    # 1. Borrower, lender and legal representative
    info, main = text.split('\n\n')
    info = subject_pat['legal_represent_replace']. \
        sub('法定代表人：', subject_pat['borrower_replace'].
            sub('被告：', subject_pat['lender_replace'].
                sub('原告：', info)))

    idx = 0
    lenders = []
    lender_pos = []
    for item in subject_pat['lender'].finditer(info):
        lenders.append(item.group(1))
        if idx:
            lender_pos.append((idx, item.span()[0]))
        idx = item.span()[1]
    lender_type = [0] * len(lenders)

    first = True
    borrowers = []
    borrower_pos = []
    for item in subject_pat['borrower'].finditer(info):
        borrowers.append(item.group(1))
        if first:
            lender_pos.append((idx, item.span()[0]))
            first = False
        else:
            borrower_pos.append((idx, item.span()[0]))
        idx = item.span()[1]
    borrower_pos.append((idx, len(info)))
    borrower_type = [0] * len(borrowers)

    represents = []
    for item in subject_pat['legal_represent'].finditer(info):
        represents.append(item.group(1))
        span = item.span()
        for idx, pos in enumerate(lender_pos):
            if span[0] > pos[0] and span[1] <= pos[1]:
                lender_type[idx] = 1
        for idx, pos in enumerate(borrower_pos):
            if span[0] >= pos[0] and span[1] <= pos[1]:
                borrower_type[idx] = 1

    for name in lenders:
        main = main.replace('原告' + name, '原告').replace(name, '原告')
    for name in borrowers:
        main = main.replace('被告' + name, '被告').replace(name, '被告')
    for name in represents:
        main = main.replace('法定代表人' + name, '法定代表人').replace(name, '法定代表人')

    lender_type = [int(not all(lender_type)), int(any(lender_type))]
    borrower_type = [int(not all(borrower_type)), int(any(borrower_type))]

    lender_count = len(lenders)
    borrower_count = len(borrowers)

    # 2. guarantee
    guarantee = [0, 0, 0]
    if '抵押' in main:
        guarantee[1] = 1
    if '担保' in main:
        guarantee[2] = 1
    if not any(guarantee):
        guarantee[0] = 1

    # 3. interest
    interest = 0
    interest_value = 0
    interest_type = [0] * 4
    for pattern, factor in interest_pat:
        for item in pattern.finditer(main):
            interest = 1
            interest_value = max(interest_value, round(float(item.group(2)) * factor, 2))
    for idx, interval in enumerate(interest_interval):
        if interest_value > interval:
            interest_type[3 - idx] = 1
            break
    if interest == 0:
        interest_type[0] = 1

    # TODO: replace the interest?

    # 4. payment methods
    payment = [0] * 4
    for idx, methods in enumerate(payment_list):
        for method in methods:
            if method in main:
                payment[idx + 1] = 1
                break
    if not any(payment):
        payment[0] = 1

    # 5. repayment
    repayment = [0] * 3
    if '已还款' in main:
        repayment[1] = 1
    if '尚未还款' in main:
        repayment[2] = 1
    if not any(repayment):
        repayment[0] = 1

    # 5. agreements
    agreement = [0] * 6
    for idx, methods in enumerate(agreement_list):
        for method in methods:
            if method in main:
                agreement[idx + 1] = 1
                break
    if guarantee[1] or guarantee[2]:
        agreement[4] = 1
    if not any(agreement):
        agreement[0] = 1

    # TODO: concat info + main ?

    return main.strip('\n'), lender_type + [lender_count] + borrower_type + [borrower_count] + guarantee + [interest] + [
        interest_value] + interest_type + payment + repayment + agreement

    # TODO: QuantileTransformer?
