"""
实现的是: 构建词典，实现方法吧句子转化成数字序列何其翻转
"""

"""
<PAD> -> 0    表示填充
<SOS> -> 1    表示开头
<EOS> -> 2    表示结尾
<UNK> -> 3    表示未知
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
        """吧单个句子保存到dict当中
        param: sentence: [word1, word2, word3...]
        """
        for word in sentence:
            self.count[word] = self.count.get(word, 0) + 1

    def build_vocab(self, min=5, max=None, max_features=None):
        """
        生成词典
        param min:          最小出现的次数
        param max:          最大的次数
        param max_features: 一共保留多少的词语
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
        
        # 得到一个翻转的dict字典
        self.reverse_dict = dict(zip(self.dict.values(), self.dict.keys()))
    
    def transform(self, sentence, max_len=None, pad_first=False):
        """
        把句子转化成序列
        param sentence: [word1, word2...]
        param max_len: int, 对句子精选填充或者裁剪
        """
        if max_len is not None: # do padding here 填充
            if pad_first == False:
                if max_len > len(sentence):
                    sentence = sentence + [self.PAD_TAG] * (max_len-len(sentence))
                if max_len < len(sentence):
                    sentence = sentence[:max_len] #裁剪
            else:
                if max_len > len(sentence):
                    sentence = [self.PAD_TAG] * (max_len-len(sentence)) + sentence
                if max_len < len(sentence):
                    sentence = sentence[-max_len:] #裁剪

        return [self.dict.get(word, self.UNK) for word in sentence]
    
    def inverse_transform(self, indices, is_tensor=False):
        """
        靶序列转化为句子
        param indices: [1, 2, 3, 4, 5...]
        """
        if is_tensor == False:
            return [self.reverse_dict.get(idx) for idx in indices]
        
        else:
            
            return [self.reverse_dict.get(idx.item()) for idx in indices]

    def __len__(self):
        return (len(self.dict))

    def save_dict(self, path):
        # Save self.dict at location path
        with open(path, 'w') as file_object:  #open the file in write mode
            json.dump(self.dict, file_object)
            
        print("Successfully Saved label dict!")

    def load_dict(self, path):
        # Load the saved dictionary to self.dict
        
        file = open(path)
        self.dict = json.load(file)
        
        self.reverse_dict = dict(zip(self.dict.values(), self.dict.keys()))
        print("Successfully Loaded label dict!")


# if __name__ == '__main__':
#     import config
#     from data import get_preprocessed_data
#     text = open(config.path).readlines()
#     seqs, segs = get_preprocessed_data(text)
#     w2s = Word2Sequence()
#     for sequence in seqs:
#         w2s.fit(sequence)
#     w2s.build_vocab(min=0, max_features=10000)
#     ret = w2s.transform(seqs[0], max_len=128, pad_first=True)
#     rev_ret = w2s.inverse_transform(ret)
#     print(ret)
#     print(rev_ret)
#     w2s.save_dict(config.w2s_dict_path)


# if __name__ == '__main__':
#     import config
#     from data import get_preprocessed_data
#     text = open(config.path).readlines()
#     seqs, segs = get_preprocessed_data(text)
#     custom_dict = {"<PAD>" : 0}
#     w2s_seg = Word2Sequence(custom_dict=custom_dict)
#     for segment in segs:
#         w2s_seg.fit(segment)
#     w2s_seg.build_vocab(min=0, max_features=12)
#     ret = w2s_seg.transform(segs[0], max_len=128, pad_first=True)
#     rev_ret = w2s_seg.inverse_transform(ret)
#     print(ret)
#     print(rev_ret)
#     w2s_seg.save_dict(config.w2s_seg_dict_path)