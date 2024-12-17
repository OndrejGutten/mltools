import numpy as np
import pandas as pd


class OCR_Evaluator():
    '''
    This class is used to evaluate the quality of OCR output.
    The evaluator is given a single text, a list of texts or a dataframe where each column represents a different OCR method. Each text is split into words, which are compared against the list of known words.
    Two metrics are produced - fraction of known words and total length of known words.
    Fraction of known words is normalized (value between 0 and 1) but will overestimates OCR methods that tend to delete unrecognized words.
    Total length of known words is the sum of lengths of all known words in the text. This is a more robust metric, but only makes sense when comparing methods that scanned the same set of texts.
    Both methods will underestimate methods that fail to recognize spaces between words.
    A list of known words has to be provided during initialization. 

    Example usage:
    curl -X POST http://127.0.0.1:5000/invocations \
     -H "Content-Type: application/json" \
     -d '{
           "dataframe_split": {
               "columns": ["method1", "method2"],
               "data": [
                   ["hello world", "Goodbye world"],
                   ["42", "New text"],
                   ["Test test test", "Another text"]
               ]
           }
         }'

    '''
    def __init__(self, known_words: list[str]):
        self.__repr__ = f"OCR Evaluator"
        self.known_words = {w.lower() for w in known_words}
        # store vocabulary
        
    def predict(self, data: str | list[str] | pd.DataFrame) -> pd.DataFrame:
        formatted_data = self._input_to_df(data)
        answer = pd.DataFrame(columns = ['method','known_fraction','known_length'])
        for col in formatted_data.columns:
            intermediate_df = pd.DataFrame({'fraction':pd.Series(dtype=float), 'known_length':pd.Series(dtype=int), 'total_length':pd.Series(dtype=int)}, index = formatted_data.index)
            column_texts_without_nan = formatted_data[col].dropna()
            intermediate_df[:] = pd.DataFrame(data = column_texts_without_nan.apply(lambda x: self._calculate_metrics_for_single_text(x)).to_list(), index = intermediate_df.index)
            fraction_micro = intermediate_df.apply(lambda x: x['fraction'] * x['total_length'], axis=1).sum() / intermediate_df['total_length'].sum()
            #fraction_macro = intermediate_df['fraction'].mean()
            known_length = intermediate_df['known_length'].sum()
            answer.loc[col] = [col, fraction_micro, known_length]
        
        return answer.astype({'method': 'str', 'known_fraction': 'float64','known_length': 'int64'})
    
    def _calculate_metrics_for_single_text(self, text):
        words_df = pd.DataFrame(data = text.lower().split(),columns = ['words'])
        words_df['known'] = [word in self.known_words for word in words_df['words']]
        words_df['length'] = words_df['words'].apply(lambda x: len(x))

        total_length = words_df['length'].sum()
        known_length = words_df['length'][words_df['known']].sum()
        fraction = known_length / total_length
        return fraction, known_length, total_length

    def _input_to_df(self, data: str | list[str] | pd.DataFrame) -> pd.DataFrame:
        if isinstance(data, str):
            mydf = pd.DataFrame([data], columns=['text'])
        if isinstance(data, list):
            mydf = pd.DataFrame(data, columns=['text']).astype(str)
        if isinstance(data, pd.DataFrame):
            mydf = data.astype(str)
        zero_length_rows = mydf.iloc[:,0].str.len() == 0
        number_of_zero_length_rows = sum(zero_length_rows)
        if number_of_zero_length_rows > 0:
            print(f"Warning: {number_of_zero_length_rows} rows have zero length and will be dropped")
            mydf = mydf[~zero_length_rows]
        
        return mydf
