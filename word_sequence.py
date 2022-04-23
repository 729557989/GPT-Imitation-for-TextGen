"""
<PAD> -> 0    padding token
<SOS> -> 1    start token
<EOS> -> 2    end token
<UNK> -> 3    unknown token
"""
import json

class Word2Sequence:
    PAD_TAG = "<PAD>"
    SOS_TAG = "<SOS>"
    EOS_TAG = "<EOS>"
    UNK_TAG = "<UNK>"

    PAD = 0
    SOS = 1
    EOS = 2
    UNK = 3
    
    special_tokens = [PAD_TAG, SOS_TAG, EOS_TAG, UNK_TAG]
        
    def __init__(self, custom_dict = None):
        self.dict = {
            self.PAD_TAG : self.PAD,
            self.SOS_TAG : self.SOS,
            self.EOS_TAG : self.EOS,
            self.UNK_TAG : self.UNK
        } if custom_dict == None else custom_dict
        
        self.count = {}

    def fit(self, sentence):
        """save words in sentence to self.dict
        param: sentence: [word1, word2, word3...]
        """
        for word in sentence:
            self.count[word] = self.count.get(word, 0) + 1

    def build_vocab(self, min=5, max=None, max_features=None):
        """
        Build Word Dictionary
        param min:          least occurrance of word to be considered
        param max:          max occurrance of word to be considered
        param max_features: max vocab size for tokenizer
        returns:            
        """
        # 删除count中词频小于min的word
        if min is not None:
            self.count = {word: value for word,value in self.count.items() if value > min}
        # 删除次数大于max的值
        if max is not None:
            self.count = {word: value for word,value in self.count.items if value < max}
        # 限制保留的词语数
        if max_features is not None:
            temp = sorted(self.count.items(), key=lambda x:x[-1], reverse=True)[:max_features]
            self.count = dict(temp)

        for word in self.count:
            if word not in self.special_tokens:
                self.dict[word] = len(self.dict)
        
        # reversed self.dict
        self.reverse_dict = dict(zip(self.dict.values(), self.dict.keys()))
    
    def transform(self, sentence, max_len=None, pad_first=False):
        """
        convert setence to int sequence
        param sentence: [word1, word2...]
        param max_len: int, do padding or truncation
        """
        if max_len is not None: # do padding here
            if pad_first == False:
                if max_len > len(sentence):
                    sentence = sentence + [self.PAD_TAG] * (max_len-len(sentence))
                if max_len < len(sentence):
                    sentence = sentence[:max_len] #truncation
            else:
                if max_len > len(sentence):
                    sentence = [self.PAD_TAG] * (max_len-len(sentence)) + sentence
                if max_len < len(sentence):
                    sentence = sentence[-max_len:] #truncation

        return [self.dict.get(word, self.UNK) for word in sentence]
    
    def inverse_transform(self, indices, is_tensor=False):
        """
        convert int sequences to words
        param indices: [1, 2, 3, 4, 5...]
        """
        if is_tensor == False:
            return [self.reverse_dict.get(idx) for idx in indices]
        
        else:
            
            return [self.reverse_dict.get(idx.item()) for idx in indices]

    def __len__(self):
        return (len(self.dict))

    def save_dict(self, path, notify=False):
        # Save self.dict at location path
        with open(path, 'w') as file_object:  #open the file in write mode
            json.dump(self.dict, file_object)
        
        if notify == True:
            print("Successfully Saved label dict!")

    def load_dict(self, path, notify=False):
        # Load the saved dictionary to self.dict
        
        file = open(path)
        self.dict = json.load(file)
        
        self.reverse_dict = dict(zip(self.dict.values(), self.dict.keys()))
        
        if notify == True:
            print("Successfully Loaded label dict!")


# if __name__ == '__main__':
#     import config
#     from data import txt_to_wordlist
#     word_list = txt_to_wordlist(config.path)
    
#     w2s = Word2Sequence()
#     w2s.fit(word_list)
#     w2s.build_vocab(min=0, max_features=None)
#     w2s.save_dict(config.w2s_dict_path)
#     print(f"This tokenizer has: {len(w2s)} vocabs")
    
#     sent = ["this", "text", "is", "for", "test"]
#     ret = w2s.transform(sent, max_len=10, pad_first=False)
#     rev_ret = w2s.inverse_transform(ret)
#     print(ret)
#     print(rev_ret)
#     w2s.save_dict(config.w2s_dict_path)