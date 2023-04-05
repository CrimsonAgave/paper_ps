import pandas as pd
import numpy as np
import re
from collections import Counter
import itertools
import gensim
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

def preprocessing(s):
    s = re.sub("[0-9]+", "0", s)

    # リンク
    s = re.sub('http:.* ', 'http ', s)
    s = re.sub('https:.* ', 'https ', s)

    # 記号
    s = re.sub(r"[\"\'.,:;\(\)#\|\*\+\!\?#$&/\]\[\{\}]", "", s)
    s = re.sub("-", " ", s)

    """
    s = re.sub("[-:,]", " ", s)
    s = re.sub(".", "", s)
    s = re.sub("\?", " \?", s)
    s = re.sub('"', ' " ', s)
    s = re.sub('\(', ' \( ', s)
    s = re.sub('\)', ' \) ', s)
    s = re.sub("\n", " ")
    s = re.sub("\r", " ")
    s = re.sub("\u3000", " ")  # 全角スペース
    """

    # 固有名詞等の他は小文字化
    if(sum(map(str.islower, s)) >= 2):
        pass
    else:
        s = s.lower()

    s = re.split(" ", s)
    s = [e for e in s if e != ""]
    return s

class Dataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # データ整形
        x = "title " + self.df.iloc[idx]["title"] +" abstract " + self.df.iloc[idx]["abstract"]
        x = preprocessing(x)
        x = get_id_list(x)
        x = torch.tensor(x)

        y = self.df.iloc[idx]["y"]
        y = torch.tensor(y)
        return x, y


def collate_fn(batch):
    sequences = [x[0] for x in batch]
    y = torch.from_numpy(np.array([x[1] for x in batch]))
    x = pad_sequence(sequences, batch_first=True, padding_value=0)
    return x, y

class Model(nn.Module):
    def __init__(self, emb_dim, hidden_dim, output_dim, padding_idx, emb_weights):
        super().__init__()
        self.hidden_dim = hidden_dim
        DROPOUT = 0.5

        self.emb = nn.Embedding.from_pretrained(
            emb_weights, padding_idx=padding_idx
        )
        self.drop1 = nn.Dropout(DROPOUT)
        self.rnn = nn.LSTM(
            input_size = emb_dim,
            hidden_size = hidden_dim,
            num_layers = 4,
            batch_first=True,
            bias=True
        )
        self.drop2 = nn.Dropout(DROPOUT)
        self.l1 = nn.Linear(
            in_features = hidden_dim,
            out_features = output_dim,
            bias = True
        )

    def forward(self, x, h_0=None):
        x = self.emb(x)
        x = self.drop1(x)
        x, h = self.rnn(x, h_0)
        x = x[:, -1, :]
        x = self.drop2(x)
        x = self.l1(x)
        return x

def train_loop(data, model, loss_fn, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0.0
    size = len(data.dataset)
    for x, y in tqdm(data):        
        optimizer.zero_grad()
        # x, y = x.to(device), y.to(device)
        outputs = model(x)
        loss = loss_fn(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        total_loss += loss.item()
        outputs = np.argmax(outputs.data.numpy(), axis=1)
        y = y.data.numpy()
        correct += (outputs == y).mean()

    total_loss /= size
    correct /= size
    return total_loss, correct

def prediction(data, model):
    model.eval()
    softmax = nn.Softmax(dim=1)
    ys = []
    with torch.no_grad():
        for x, _ in tqdm(data):
            y = model(x)
            # y = int(y.argmax())
            ys.append(y)
    return ys


# データ読み込み
print("データ読み込み開始")
emb_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

train_filename = "train_data.csv"
df_train = pd.read_csv(train_filename, index_col="id")
test_filename = "test_data.csv"
df_test = pd.read_csv(test_filename, index_col="id")
df_test['y'] = 0
print("データ読み込み終了")

# text2id
train_text = df_train["title"].str.cat(sep="") + df_train["abstract"].str.cat(sep="")
train_words = preprocessing(train_text)
words_freq = Counter(itertools.chain(train_words))

word_to_id = {}
for i, word_uniq in enumerate(words_freq.most_common(), start=1):
    if(word_uniq[1] < 2): 
        break
    word_to_id.update({word_uniq[0]: i})

def get_id_list(words):
    ids = []
    for word in words:
        if(word in word_to_id.keys()):
            id = word_to_id[word]
        else:
            id = 0
        ids.append(id)
    return ids


BATCH_SIZE = 16
train_dataset = Dataset(df_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
test_dataset = Dataset(df_test)
test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)

VOCAB_SIZE = len(word_to_id)+1
PADDING_IDX = 0
EMB_DIM = 300
HIDDEN_DIM = 2
OUTPUT_DIM = 2
LEARNING_RATE = 1e-5
EPOCH_SIZE = 10

# 学習済み単語ベクトルの取得
weights = np.zeros((VOCAB_SIZE, EMB_DIM))
words_in_pretrained = 0
for i, word in enumerate(word_to_id.keys()):
    try:
        weights[i] = emb_model[word]
        words_in_pretrained += 1
    except KeyError:
        weights[i] = np.random.normal(scale=0.1, size=(EMB_DIM,))
weights = torch.from_numpy(weights.astype((np.float32)))

model = Model(emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, padding_idx=PADDING_IDX, emb_weights=weights)
loss_fn = nn.CrossEntropyLoss()
from lion_pytorch import Lion
optimizer = Lion(model.parameters(), lr=LEARNING_RATE)

# 学習
train_loss_history = []
train_acc_history = []
for i in range(EPOCH_SIZE):
    print("Epoch:", i+1, "--------------------------")
    loss, acc = train_loop(train_loader, model, loss_fn, optimizer)
    print("train_acc: ", acc)
    train_loss_history.append(loss)
    train_acc_history.append(acc)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].plot(train_loss_history)
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("loss")
axes[1].plot(train_acc_history)
axes[1].set_xlabel("epoch", size=14)
axes[1].set_ylabel("acc", size=14)
fig.savefig("test.png")

# 予測
y_test = prediction(test_loader, model)

# データ整形
df_submit = pd.DataFrame({"y": y_test})
df_submit.to_csv("submission.csv")