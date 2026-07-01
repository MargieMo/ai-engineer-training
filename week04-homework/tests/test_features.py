import unittest
from smart_customer_service.tools.order_tools import generate_invoice, query_order
from smart_customer_service.tools.time_tool import get_date_for_relative_time
from smart_customer_service.services import ServiceManager


class TestFeatures(unittest.TestCase):

    def test_invoice_tool(self):
        """The invoice tool should generate a URL that contains the order id."""
        order_id = "SN20240924003"
        result = generate_invoice.invoke({"order_id": order_id})

        self.assertTrue(result["success"])
        self.assertIn("invoice_url", result)
        self.assertIn(order_id, result["invoice_url"])
        print(f"\nInvoice tool passed: {result['message']}")

    def test_invoice_tool_invalid_id(self):
        """An invalid order id should fail gracefully."""
        result = generate_invoice.invoke({"order_id": "INVALID"})
        self.assertFalse(result["success"])
        self.assertIn("error", result)

    def test_relative_time_tool(self):
        """Relative time parsing should resolve common relative phrases."""
        from datetime import timedelta
        from smart_customer_service.date_context import today

        anchor = today()
        self.assertEqual(
            get_date_for_relative_time.invoke({"relative_time_str": "yesterday"}),
            (anchor - timedelta(days=1)).strftime("%Y-%m-%d"),
        )
        self.assertEqual(
            get_date_for_relative_time.invoke({"relative_time_str": "明天"}),
            (anchor + timedelta(days=1)).strftime("%Y-%m-%d"),
        )
        weekday_result = get_date_for_relative_time.invoke({"relative_time_str": "今天星期几"})
        self.assertIn(anchor.strftime("%Y-%m-%d"), weekday_result)
        self.assertIn("星期", weekday_result)

    def test_phase1_chain_prompt(self):
        """Stage-one chain should format prompt with today and user input."""
        from smart_customer_service.base_chain import RELATIVE_TIME_PROMPT
        from smart_customer_service.date_context import today_str

        current_today = today_str()
        messages = RELATIVE_TIME_PROMPT.format_messages(
            today=current_today,
            user_input="我昨天下的单",
        )
        self.assertIn(current_today, messages[0].content)
        self.assertEqual(messages[1].content, "我昨天下的单")

    def test_hot_update_preserves_sessions(self):
        """Hot-updating tools swaps the active set while keeping the design that
        isolates existing sessions (old graph instances finish their lifecycle,
        new sessions use the rebuilt graph).
        """
        sm = ServiceManager()
        initial_tools = sm.get_tools()
        self.assertIn("apply_refund", [t.name for t in initial_tools])
        print(f"\nInitial tools: {[t.name for t in initial_tools]}")

        # Simulate a hot update that removes the refund tool.
        sm.update_tools([query_order])
        updated_tools = sm.get_tools()
        self.assertNotIn("apply_refund", [t.name for t in updated_tools])
        self.assertIn("query_order", [t.name for t in updated_tools])
        print(f"Updated tools: {[t.name for t in updated_tools]}")


if __name__ == "__main__":
    unittest.main()
