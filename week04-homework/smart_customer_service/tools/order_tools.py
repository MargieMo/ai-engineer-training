import random
import time
from langchain.tools import tool


# In-memory mock database. In a real system this would be a call to an order service.
_MOCK_ORDER_DB = {
    "SN20240924001": {"status": "Shipped", "tracking_number": "SF123456789", "items": ["LangChain Starter T-shirt"]},
    "SN20240925001": {"status": "Shipped", "tracking_number": "SF987654321", "items": ["AI Agent Developer Mug"]},
    "SN20240924002": {"status": "Pending Payment", "tracking_number": None, "items": ["LangGraph Advanced Sticker"]},
    "SN20240924003": {"status": "Completed", "tracking_number": "JD987654321", "items": ["AI Agent Developer Mug"]},
}


@tool
def query_order(order_id: str) -> dict:
    """Look up the status and shipping information of an order by its order id.
    Call this tool whenever the user wants to check an order.
    """
    print(f"--- [Tool Call] Querying order id: {order_id} ---")
    time.sleep(2)  # Simulate database / network latency.
    order_info = _MOCK_ORDER_DB.get(order_id)
    if order_info:
        return {
            "success": True,
            "order_id": order_id,
            "status": order_info["status"],
            "tracking_number": order_info["tracking_number"],
            "details": f"Items in the order: {', '.join(order_info['items'])}",
        }
    return {
        "success": False,
        "order_id": order_id,
        "error": "Order not found. Please double-check the order id.",
    }


@tool
def apply_refund(order_id: str, reason: str) -> dict:
    """Apply for a refund for the order with the given order id.
    Requires the order id and a refund reason.
    """
    print(f"--- [Tool Call] Applying refund for order {order_id}, reason: {reason} ---")
    time.sleep(1)  # Simulate processing latency.
    if "SN" in order_id:
        refund_id = f"REFUND_{random.randint(1000, 9999)}"
        return {
            "success": True,
            "order_id": order_id,
            "refund_id": refund_id,
            "message": "Refund request submitted. Once approved, the amount will be returned to the original payment method.",
        }
    return {
        "success": False,
        "order_id": order_id,
        "error": "Invalid order id, unable to apply for a refund.",
    }


@tool
def generate_invoice(order_id: str) -> dict:
    """Generate an invoice for the order with the given order id.
    Call this tool when the user needs an invoice.
    """
    print(f"--- [Tool Call] Generating invoice for order {order_id} ---")
    time.sleep(1)  # Simulate processing latency.
    if "SN" in order_id:
        invoice_url = f"https://example.com/invoices/{order_id}.pdf"
        return {
            "success": True,
            "order_id": order_id,
            "invoice_url": invoice_url,
            "message": f"Invoice generated. You can download it here: {invoice_url}",
        }
    return {
        "success": False,
        "order_id": order_id,
        "error": "Invalid order id, unable to generate an invoice.",
    }
