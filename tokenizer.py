import pandas as pd 
import numpy as np
import re
from collections import Counter
import pickle

class SyllableTokenizer:
    """
    A class for tokenizing text into syllables.
    """

    def __init__(self, data):
        """
        Initialize the SyllableTokenizer object.

        :param data: The text data to tokenize into syllables.
        :type data: str
        """

        self.speller = Speller()
        self.vocabulary = {}
        self.text_data = data
        # self.data = sorted([i for i in set(data)])
        self.idxtos = {}
        # self.idxtos = {idx:w for idx,w in enumerate(self.data)}
        # self.stoidx = {w:idx for idx,w in enumerate(self.data)}
        self.stoidx = {}
        # self.decode = lambda i: "".join(self.idxtos[j] for j in i)
    
    def decode(self, indices):
        """
        Decode a sequence of indices into syllables.

        :param indices: The sequence of indices to decode.
        :type indices: list[int]
        :return: The decoded syllables.
        :rtype: str
        """

        return "".join(self.idxtos[j] for j in indices)

    
    def getSyllableVocab(self):
        """
        Extract the syllable vocabulary from the text data.
        """

        pattern = r'[aqwertyuıopğüişlkjhgfdsazxcvbnmöçQWERTYUIOPĞÜİŞLKJHGFDSAZXCVBNMÖÇÂâûÛîÎ̉̉]'
        self.syllables = []
        errors = []
        spellable_words = re.findall(r'\b\w+' + pattern + r'\w*\b', self.text_data)
        for word in spellable_words:
            try:
                # print(speller.spell(word))
                self.syllables.extend(speller.spell(word))
                # self.syllables.append(" ")
            except:
                print("Word: ",word," couldn't be tokenized...")
                print("<=>"*57)
                errors.append(word)
                pass
        self.spellable_words = [x for x in spellable_words if x not in errors]
        
        for idx, w in enumerate(sorted(set(self.syllables))):
            if w not in self.stoidx:
                self.idxtos[1 + len(self.idxtos)] = w
                self.stoidx[w] = 1 + len(self.stoidx)  
    def trainSyllableVocab(self, timestep):
        """
        Train the syllable vocabulary based on the text data.

        :param timestep: The number of training iterations.
        :type timestep: int
        """
        self.merges = Counter()
        self.find_merge = {}
        syllables_encoded = [self.stoidx[i] for i in self.syllables]
        for time in range(timestep):
            counts = Counter()
            for idx in range(len(syllables_encoded) -1):
                candidate = (syllables_encoded[idx],syllables_encoded[idx + 1])

                counts[candidate] += 1
            
            max_pair = max(counts, key=counts.get)
            self.merges[max_pair] += counts[max_pair]

            self.find_merge[(max_pair[0], max_pair[1])] = 1 + len(self.idxtos)
            self.idxtos[1 + len(self.idxtos)] = self.idxtos[max_pair[0]] + self.idxtos[max_pair[1]]
            self.stoidx[self.idxtos[max_pair[0]] + self.idxtos[max_pair[1]]] = 1 + len(self.stoidx)

            print(max_pair, "is going to be =========>", self.stoidx[self.idxtos[max_pair[0]] + self.idxtos[max_pair[1]]],f"       {time}/{timestep}")
            print(self.idxtos[max_pair[0]]," + ",self.idxtos[max_pair[1]], "is going to be =========>", self.idxtos[max_pair[0]] + self.idxtos[max_pair[1]])
            syllables_encoded = self.update(max_pair[0], max_pair[1],self.stoidx[self.idxtos[max_pair[0]] + self.idxtos[max_pair[1]]] ,syllables_encoded)

    def trainNonSyllableVocab(self):
        """
        Train the vocabulary for non-syllable tokens.
        """
        pass

    def save(self, filename):
        """
        Save the SyllableTokenizer object to a file.

        :param filename: The filename to save the object to.
        :type filename: str
        """

        with open(filename, 'wb') as file:
            pickle.dump(self, file)
    
    @staticmethod
    def from_pretrained(filename):
        """
        Load a SyllableTokenizer object from a file.

        :param filename: The filename from which to load the object.
        :type filename: str
        :return: The loaded SyllableTokenizer object.
        :rtype: SyllableTokenizer
        """
        with open(filename, 'rb') as file:
            return pickle.load(file)


    def update(self, idx1, idx2, target, old_list):
        """
        Update a list by replacing occurrences of `idx1` followed by `idx2` with `target`.

        :param idx1: The first index to replace.
        :type idx1: int
        :param idx2: The second index to replace.
        :type idx2: int
        :param target: The value to replace occurrences of `idx1` followed by `idx2`.
        :type target: any
        :param old_list: The original list to be updated.
        :type old_list: list
        :return: The updated list with replacements made.
        :rtype: list
        """
        n = 0
        new_list = []
        while n < len(old_list) - 1:  # Modified loop condition
            left = old_list[n]
            right = old_list[n + 1]

            if left == idx1 and right == idx2:
                new_list.append(target)
                n += 2
            else:
                new_list.append(left)
                n += 1

        # Append the last element if there's any
        if n == len(old_list) - 1:
            new_list.append(old_list[-1])

        return new_list
    



    def encode(self, sentence):
        """
        Encode a sentence into a sequence of indices representing syllables.

        :param sentence: The sentence to encode.
        :type sentence: str
        :return: The encoded sequence of indices.
        :rtype: list[int]
        """
        sentence = sentence.split()
        sentence = [self.speller.spell(word) for word in sentence]
        encoded_list = [self.stoidx[char] for word in sentence for char in word]
        counts = Counter()
        check_merges = True

        while check_merges:
            check_merges = False
            for idx in range(len(encoded_list) - 1):
                candidate = (encoded_list[idx], encoded_list[idx + 1])

                if candidate in self.merges:
                    check_merges = True
                    counts[candidate] += self.merges[candidate]

            if counts:  # Check if counts is not empty
                max_pair = max(counts, key=counts.get)
                encoded_list = self.update(max_pair[0], max_pair[1], self.find_merge[max_pair],encoded_list)
        return encoded_list
        


class Checker:
    """
    A class for checking whether a given letter is a vowel or a consonant.
    """

    def __init__(self, language):
        """
        Initialize the Checker object.

        :param language: The language for which the Checker object is instantiated.
        :type language: str
        """

        self.language = language

    def isVowel(self,letter):
        """
        Check if a given letter is a vowel.

        :param letter: The letter to check.
        :type letter: str
        :return: True if the letter is a vowel, False otherwise.
        :rtype: bool
        """

        if letter in "aäáâeıioôÔöuüAÁÂÄEIîÎİOÖUÜ":
            return True
        return False

    def isConsonant(self,letter):
        """
        Check if a given letter is a consonant.

        :param letter: The letter to check.
        :type letter: str
        :return: True if the letter is a consonant, False otherwise.
        :rtype: bool
        """
        if letter in "bcçdfgğhjklmnprsştvyzwqxBCÇDFGĞHJKLMNPRSŞTVYZWQX":
            return True
        return False

    def transformCV(self,text):
        """
        Transform the input text into a sequence of 'C's, 'V's, or 'P's,
        representing consonants, vowels, or punctuation, respectively.

        :param text: The input text to transform.
        :type text: str
        :return: The transformed text.
        :rtype: str
        """
        transformed_list = []

        for letter in text:
            if self.isConsonant(letter):
                transformed_list.append("C")
            elif self.isVowel(letter):
                transformed_list.append("V")
            elif letter == " ":
                transformed_list.append(" ")
            else:
                transformed_list.append("P")
            
        return "".join(transformed_list)


class Speller:
    """
    A class for spelling words based on syllables.
    """

    def __init__(self, language="Turkish"):
        """
        Initialize the Speller object.

        :param language: The language for which the Speller object is instantiated. Default is "Turkish".
        :type language: str
        """
        self.checker = Checker(language)
    

    def findFirstSyllable(self,word):
        """
        Find the first syllable of a word.

        :param word: The word for which to find the first syllable.
        :type word: str
        :return: The first syllable of the word.
        :rtype: str
        """
        # if all(char.isupper() for char in word):
        #     # return [char for char in word][0]
        #     pass

        if len(word) == 1:
            return word
        
        if self.checker.isConsonant(word[0]):
            if self.checker.isConsonant(word[1]):
                return word[0]


            else:
                if len(word) == 2:
                    return word[:2]
                if self.checker.isConsonant(word[2]):
                    if len(word) == 3:
                        return word[:3]
                    if self.checker.isConsonant(word[3]):
                        if len(word)>4 and self.checker.isConsonant(word[4]):
                            return word[:4]
                        else:
                            return word[:3]
                    else:
                        return word[:2]
                else:
                    return word[:2]
      
        if self.checker.isVowel(word[0]):
            if len(word) < 3 or len(word) < 4:
                return word
            if self.checker.isConsonant(word[2]):
                if self.checker.isConsonant(word[3]):
                    return word[:3]
                else:
                    return word[:2]
            else:
                return word[0]
            
    

    def spell(self,word):
        """
        Spell a word based on its syllables.

        :param word: The word to spell.
        :type word: str
        :return: A list of syllables that make up the word.
        :rtype: list[str]
        """
        spell_list = []

        while len(word) > 0:
            first_syllable = self.findFirstSyllable(word)
            word = word[len(first_syllable):]
            spell_list.append(first_syllable)

        return spell_list
