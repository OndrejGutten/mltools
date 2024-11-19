from tqdm import tqdm
import os
import regex
from transformers import AutoTokenizer
import json
from nltk import word_tokenize
import string
import simplemma
import re
import logging
import nltk
import pandas as pd
from mltools import utils

# TODO: replace stopwords with mask tokens

nltk.download('punkt')


class TextPreprocessor:
    '''
    Class used for removing stop words, removing repeated characters, and despacing text

    Arguments:
    input_dir -- path to original data
    output_dir -- path to preprocessed data
    stop_words_file -- file containing stop words in JSON format
    repeated_characters_threshold -- threshold for removing repeated characters
    despacing_primary_vocabulary -- csv file (spaced with " ") containing valid words in the 2nd column
    '''

    def __init__(self, **kwargs):

        tokenizer_hf_id = kwargs.get('tokenizer_hf_id', None)
        if (tokenizer_hf_id is not None):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_hf_id)

        self.window_overlap = kwargs.get('window_overlap')
        self.stop_words_file = kwargs.get("stop_words_file")
        self.repeated_characters_threshold = kwargs.get(
            "repeated_characters_threshold")
        self.despacing_vocabulary = kwargs.get("despacing_vocabulary")
        self.chunk_size = kwargs.get("chunk_size")
        self.callable_functions = {
            'remove_gender_symmetry': self.remove_gender_symmetry,
            'insert_spaces_around_punctuation': self.insert_spaces_around_punctuation,
            'lemmatize': self.lemmatize,
            'chunkify': self.chunkify,
            'remove_stop_words': self.remove_stop_words,
            'remove_repeated_characters': self.remove_repeated_characters,
            'despace': self.despace,
            'replace_email': self.replace_email,
            'replace_rodne_cislo': self.replace_rodne_cislo,
            'replace_IC': self.replace_IC,
            'replace_dates': self.replace_dates,
            'replace_urls': self.replace_urls,
            'replace_formatted_numbers': self.replace_formatted_numbers,
            'remove_non_ascii': self.remove_non_ascii,
            'remove_placeholders': self.remove_placeholders,
            'replace_numbers': self.remove_numbers,
            'remove_strings_with_numbers_or_punctuation': self.remove_strings_with_numbers_or_punctuation,
            'split_to_sentences': self.split_to_sentences,
            'remove_unknown_words': self.remove_unknown_words,
            'replace_unknown_words': self.replace_unknown_words,
            # 'match_unknown_words': self.match_unknown_words
        }

    def preprocess_single_file(self, input_file, output_file, preprocess_stack: list[str]):
        text = self.get_text(input_file)
        preprocessed_text = self.preprocess_single(text, preprocess_stack)
        with open(output_file, "w") as f:
            if isinstance(preprocessed_text, list):
                preprocessed_text = json.dumps(preprocessed_text)
            f.write(preprocessed_text)

    def preprocess_single(self, text, preprocess_stack: list[str]):
        functions_to_call = [self.callable_functions.get(function, None)
                             for function in preprocess_stack]
        not_found = [f for index, f in enumerate(
            preprocess_stack) if functions_to_call[index] is None]
        for f in not_found:
            logging.warning(f"Function {f} not found")
        functions_to_call = [f for f in functions_to_call if f is not None]
        for preprocess_function in functions_to_call:
            text = preprocess_function(text)
        return text

    def preprocess_iterable(self, iterable, preprocess_stack: list[str]) -> list[str]:
        return [self.preprocess_single(text, preprocess_stack) for text in utils.make_iterable(iterable)]

    def preprocess_folder(self, input_dir: str, output_dir: str, preprocess_stack: list[str]):
        for p in tqdm(sorted(os.listdir(input_dir))):
            os.makedirs(f"{output_dir}/{p}", exist_ok=True)
            input_file = os.path.join(input_dir, p, "Text.txt")
            output_file = os.path.join(output_dir, p, "Text.txt")
            self.preprocess_single_file(
                input_file, output_file, preprocess_stack)

    def get_text(self, file):
        with open(file, mode='r', errors='ignore') as f:
            return f.read()

    def remove_gender_symmetry(self, text):
        regex_string_a = r'/[aá]'
        regex_string_ka = r'/ka'
        regex_string_ke = r'/ke'
        regex_string_ku = r'/ku'
        replaced_text = regex.sub(regex_string_a, '', text)
        replaced_text = regex.sub(regex_string_ka, '', replaced_text)
        replaced_text = regex.sub(regex_string_ke, '', replaced_text)
        replaced_text = regex.sub(regex_string_ku, '', replaced_text)
        return replaced_text

    def insert_spaces_around_punctuation(self, text):
        punctuation_chars = './"\'`;,!?'

        # Add a space before punctuation if not already present
        pattern_before = fr'(?<!\s)([{punctuation_chars}])'
        substitution_before = r' \1'

        # Add a space after punctuation if not already present
        pattern_after = fr'([{punctuation_chars}])(?!\s)'
        substitution_after = r'\1 '

        # Apply the regex substitution
        result = re.sub(pattern_before, substitution_before, text)
        result = re.sub(pattern_after, substitution_after, result)

        # Remove any multiple spaces that might have been added
        return re.sub(r'\s+', ' ', result)

    def lemmatize(self, text):
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in string.punctuation]
        lemmas = [simplemma.lemmatize(t, lang="sk") for t in tokens]
        return " ".join(lemmas)

    def split_to_sentences(self, text):
        sentences = nltk.tokenize.sent_tokenize(text, language='czech')
        return sentences if len(sentences) > 0 else [" "]

    def replace_dates(self, text):
        return re.sub(r'\d{1,2}\.[ ]*\d{1,2}\.[ ]*\d{4}', ' [DATE] ', text)

    def replace_urls(self, text):
        text = re.sub(r'(http|https)://[^\s]*', ' [URL] ', text)
        return re.sub(r'www\.[^\s]*', ' [URL] ', text)

    def replace_formatted_numbers(self, text):
        return re.sub(r'(\d+([,]\d+)*\s*)+', ' [NUMBER] ', text)

    def remove_non_ascii(self, text):
        return regex.sub(r'[^\w\s\p{L}\p{M}\p{P}]', '', text)

    def remove_placeholders(self, text):
        return re.sub(r'\[[a-zA-Z ]*\]', '', text)

    def remove_numbers(self, text):
        return re.sub(r' \d+ ', ' [NUMBER] ', text)
        pass

    def remove_strings_with_numbers_or_punctuation(self, text):
        # Replace 'words' that contain both letters+numbers or letters+unusual punctuation with [NONWORD]
        # Also, concatenate single-letter words together

        nonword_punctuation = '"#$%&\'()*+-/<=>@[\\]^_`{|}~'

        def replace_nonword(word):
            if any(c.isdigit() for c in word) and any(c.isalpha() for c in word) and not bool(re.search(r'\[[a-zA-Z ]*\]', word)):
                return '[NONWORD]'
            if any(c.isalpha() for c in word) and any(c in nonword_punctuation for c in word) and not bool(re.match(r'\[[a-zA-Z ]*\]', word)):
                return '[NONWORD]'
            return word

        bow = text.split(' ')

        # for each word, if it contains both letters and numbers replace it with NONWORD
        # if it contains letters and punctuation replace it with NONWORD
        replaced_text = ' '.join([replace_nonword(word)
                                 for word in bow])

        return replaced_text

    def chunkify(self, reasoning: str):
        def _binary_search_idx(tokens, left_idx, right_idx):
            # this will still fail if a "word" is a long list of characters
            if left_idx > right_idx:  # If we overshoot the search space
                return right_idx  # Return the highest valid index found so far

            mid_idx = (left_idx + right_idx) // 2
            chunk_tokens = tokens[:mid_idx + 1]
            num_tokens = len(self.tokenizer(
                " ".join(chunk_tokens))["input_ids"])

            if num_tokens == self.chunk_size:
                # If we hit the exact window size, this is the best case.
                return mid_idx + 1
            elif num_tokens < self.chunk_size:
                # If the number of tokens is less than the window size, try to find a larger segment.
                return _binary_search_idx(tokens, mid_idx + 1, right_idx)
            else:
                # If the number of tokens exceeds the window size, reduce the segment size.
                return _binary_search_idx(tokens, left_idx, mid_idx - 1)

        all_tokens = [t for t in reasoning.split() if len(t.strip()) > 0]
        chunk_tokens = []
        chunks = []
        while len(all_tokens) > 0:
            all_tokens = chunk_tokens + all_tokens
            index = _binary_search_idx(all_tokens, 0, len(all_tokens) - 1)
            chunk_tokens = all_tokens[:(index + 1)]
            all_tokens = all_tokens[(index + 1):]
            chunks.append(" ".join(chunk_tokens))

            overlap_size = int(len(chunk_tokens) * self.window_overlap)
            if (overlap_size == 0):
                overlap_size = 1
            chunk_tokens = chunk_tokens[-overlap_size:]

        return chunks if len(chunks) > 0 else [" "]

    def remove_stop_words(self, text):
        with open(self.stop_words_file, 'r') as f:
            stop_words = json.load(f)

        return (" ").join([w for w in text.split() if w not in stop_words])

    def remove_repeated_characters(self, text):
        regex_pattern = r'(.)\1{' + \
            str(self.repeated_characters_threshold) + r',}'
        return re.sub(regex_pattern, lambda match: match.group(1) * 5, text)

    def read_vocabulary(self, vocab_file):
        vocabulary = set()
        with open(vocab_file, 'r') as file:
            for line in file:
                parts = line.strip().split('\t')  # Adjust delimiter based on your file format
                if len(parts) > 1:
                    # Assumes vocab is in the second column
                    vocabulary.add(parts[1].strip().upper())
                else:
                    print(line)
        return vocabulary

    def despace(self, text):
        # Regex to match sequences of spaced letters including a possible trailing space
        spaced_word_pattern = regex.compile(
            r'(?:^|\s)(\p{Lu}\s)+\p{Lu}([\s:\.;,])', flags=regex.IGNORECASE)

        def match_to_words(match):
            # Remove all spaces and get the potential word sequence
            spaced_sequence = match.group().replace(" ", "").upper()
            # Check for maximum matching words in the vocabulary
            words = []
            i = 0
            while i < len(spaced_sequence):
                found = False
                # Check for the longest possible word from this point
                for j in range(len(spaced_sequence), i, -1):
                    potential_word = spaced_sequence[i:j]
                    if potential_word in self.despacing_vocabulary:
                        words.append(potential_word)
                        i = j
                        found = True
                        break
                if not found:
                    words.append(spaced_sequence[i])
                    i += 1
            return ' ' + ' '.join(words).lower() + match.group(1)

        # Replace the matches in the original text
        despaced_words = spaced_word_pattern.sub(
            lambda m: match_to_words(m), text)

        bow = despaced_words.split(' ')
        concatenated_bow = []
        concatenated_word = ''
        for word in bow:
            if len(word) == 1:
                concatenated_word = concatenated_word + word
            elif len(concatenated_word) > 0:
                concatenated_bow.append(concatenated_word)
                concatenated_word = ''
                concatenated_bow.append(word)
            else:
                concatenated_bow.append(word)

        return ' '.join(concatenated_bow)

    def replace_email(self, text):
        regex_string = '(?<= )([a-zA-Z0-9._%-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})(?!a-zA-Z0-9)'
        replacement = '[e-mail]'
        return regex.sub(regex_string, replacement, text)

    def replace_rodne_cislo(self, text):
        regex_string = '(?<= )(\d{6}/\d{4})(?!\d)'
        replacement = '[rodne_cislo]'
        return regex.sub(regex_string, replacement, text)

    def replace_IC(self, text):
        regex_string = '(?<= )(SK|CZ)?(\d{8,10})(?!\d)'
        replacement = '[IC]'
        return regex.sub(regex_string, replacement, text)

    def remove_unknown_words(self, text, lang='sk'):
        self.replace_unknown_words(text, lang, '')

    def replace_unknown_words(self, text, lang='sk', replacement='[UNKWORD]'):
        words = text.split()
        replaced_words = []
        for w in words:
            # if the word contains any numbers, skip it
            if any(c.isdigit() for c in w):
                replaced_words.append(w)
                continue
            # if word contains punctuation - skip it (oversimplified)
            if any(c in string.punctuation for c in w):
                replaced_words.append(w)
                continue
            # if the word contains a placeholder - skip it
            if '[' in w and ']' in w:
                replaced_words.append(w)
                continue
            # if the word contains any punctuation, replace the replace the characters with a placeholder
            replaced_words.append(
                self._replace_word_with_string_if_unknown(w, replacement, lang))
        return ' '.join(replaced_words)

    def match_unknown_words(self, text, lang='sk'):
        pass

    def _replace_word_with_string_if_unknown(self, word, replacement, lang='sk'):
        # Match leading punctuation, the main word part, and trailing punctuation
        match = re.match(r"([^\w]*)(\w*)([^\w]*)", word)

        leading_punct = match.group(1)
        word_part = match.group(2)
        trailing_punct = match.group(3)

        if word_part == '':
            return word
        if simplemma.is_known(word_part, lang):
            final_string = word_part
        else:
            final_string = replacement

        return f"{leading_punct}{final_string}{trailing_punct}"
