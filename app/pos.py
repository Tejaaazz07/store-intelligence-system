import json

def get_pos_metrics():

    with open("sample_data/pos_transaction.json", "r") as f:
        transactions = json.load(f)

    total_transactions = len(transactions)

    total_revenue = sum(
        t["amount"] for t in transactions
    )

    avg_basket_value = (
        total_revenue / total_transactions
        if total_transactions > 0
        else 0
    )

    return {
        "transactions": total_transactions,
        "revenue": total_revenue,
        "avg_basket_value": round(avg_basket_value, 2)
    }
