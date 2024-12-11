import pytest
import pandas as pd
import mltools
from mltools.architecture.OCR_Evaluator import OCR_Evaluator

@pytest.fixture
def known_words():
    return ["hello", "world", "test", "goodbye", "new", "text", "another"]

@pytest.fixture
def evaluator(known_words):
    return OCR_Evaluator(known_words)

def test_predict_with_string(evaluator):
    input_data = "hello wordl"
    result = evaluator.predict(input_data)
    expected = pd.DataFrame({'method': ['text'], 'known_fraction': [0.5], 'known_length': [5]}, index=['text'])
    pd.testing.assert_frame_equal(result, expected)

def test_predict_with_list_of_strings(evaluator):
    input_data = ["hello world", "new text"]
    result = evaluator.predict(input_data)
    expected = pd.DataFrame({'method': ['text'],'known_fraction': [1.0], 'known_length': [17]}, index=['text'])
    pd.testing.assert_frame_equal(result, expected)

def test_predict_with_dataframe(evaluator):
    input_data = pd.DataFrame({
        'method1': ["hello world", "test test test"],
        'method2': ["goodbye world", "another text"]
    })
    result = evaluator.predict(input_data)
    expected = pd.DataFrame({
        'method': ['method1', 'method2'],
        'known_fraction': [1.0, 1.0],
        'known_length': [22, 23]
    }, index=['method1', 'method2'])
    pd.testing.assert_frame_equal(result, expected)