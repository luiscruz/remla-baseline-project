"""Module to test ML pipeline logic"""
import os
import unittest

import pandas as pd

from src.preprocess.preprocess_data import (  # pylint: disable=no-name-in-module,import-error
    text_prepare,
)
from src.scraping_service.app.data_validation import remove_anomalies

os.environ["PROMETHEUS_MULTIPROC_DIR"] = "/tmp/prom"  # nosec


class TestPipeLine(unittest.TestCase):
    """Unit tests for the ML pipeline"""

    def test_text_prepare(self):

        """Test text preparation"""
        examples = [
            "SQL Server - any equivalent of Excel's CHOOSE function?",
            "How to free c++ memory vector<int> * arr?",
        ]
        answers = ["sql server equivalent excels choose function", "free c++ memory vectorint arr"]
        check = False
        wrong_case = None
        for ex, ans in zip(examples, answers):
            if text_prepare(ex) != ans:
                check = True
                wrong_case = ex

        message = f"Wrong answer for the case: '{wrong_case}'"
        self.assertFalse(check, message)

    def test_data_validation(self):
        """Test data validation/cleaning"""
        df = pd.DataFrame(
            {
                "title": ["this is a test title", "test 2", "test empty"],
                "tags": [["test", "tags", "one", "python"], ["test tags two"], []],
            }
        )
        num_removed, update_df = remove_anomalies(df, valid_tags={"python"})
        self.assertEqual(num_removed, 2, msg="Expected 2 entries to be removed, but num removed was not 2")
        self.assertEqual(len(update_df), 1, msg="Output df length was not 2")
        self.assertIn("this is a test title", update_df.title)
