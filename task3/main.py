import torch
from torch import nn, optim
import torch.nn.functional as F
from torchtext import data, datasets
import numpy as np


#print('GPU:', torch.cuda.is_available())

torch.manual_seed(123)  # 为CPU中设置种子，生成随机数
# 设置随机种子可以确保每次生成固定的随机数，这就使得每次实验结果显示一致了，有利于实验的比较和改进。


'''载入数据'''
TEXT = data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm')
LABEL = data.LabelField(dtype=torch.float)
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

#print('len of train data:', len(train_data))  # len of train data: 25000
#print('len of test data:', len(test_data))  # len of test data: 25000

#print(train_data.examples[15].text)
#print(train_data.examples[15].label)

# 对文本内容进行编码
TEXT.build_vocab(train_data, max_size=10000, vectors='glove.6B.100d')
LABEL.build_vocab(train_data)


batchsz = 16
device = torch.device('cuda')
train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data),
    batch_size=batchsz,
    device=device
)


class MyLSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        """
        """
        super(MyLSTM, self).__init__()

        # [0-10001] => [100]
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # 对单词进行编码。
        # 编码vocab_size个单词，每个单词编码为embedding_dim维度的向量
        # [100] => [256]
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=2,
                           bidirectional=True, dropout=0.5)  # memory是hidden_dim
        # [256*2] => [1]
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        x: [seq_len, b] vs [b, 3, 28, 28]
        """
        # [seq, b, 1] => [seq, b, 100]
        embedding = self.dropout(self.embedding(x))

        # output: [seq, b, hid_dim*2]
        # hidden/h: [num_layers*2, b, hid_dim]
        # cell/c: [num_layers*2, b, hid_di]
        output, (hidden, cell) = self.rnn(embedding)

        # [num_layers*2, b, hid_dim] => 2 of [b, hid_dim] => [b, hid_dim*2]
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)

        # [b, hid_dim*2] => [b, 1]
        hidden = self.dropout(hidden)
        out = self.fc(hidden)

        return out


# 定义GRU网络

class GRUNet(nn.Module):
    def __init__(self, vocab_size,embedding_dim, hidden_dim, layer_dim, output_dim):
        """
        vocab_size:词典长度
        embedding_dim:词向量的维度
        hidden_dim: GRU神经元个数
        layer_dim: GRU的层数
        output_dim:隐藏层输出的维度(分类的数量)
        """
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim ## GRU神经元个数
        self.layer_dim = layer_dim ## GRU的层数
        ## 对文本进行词项量处理
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM ＋ 全连接层
        self.gru = nn.GRU(embedding_dim, hidden_dim, layer_dim,
                          batch_first=True)
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        embeds = self.embedding(x)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        r_out, h_n = self.gru(embeds, None)   # None 表示初始的 hidden state 为0
        # 选取最后一个时间点的out输出
        out = self.fc1(r_out[:, -1, :])
        return out


'''构建网络'''
LSTM = MyLSTM(len(TEXT.vocab), 100, 256)
#GRU = GRUNet(len(TEXT.vocab), 100, 256, 1, 1)
state_dict = torch.load('net_params.pth')
LSTM.load_state_dict(state_dict)

pretrained_embedding = TEXT.vocab.vectors
print('pretrained_embedding:', pretrained_embedding.shape)
LSTM.embedding.weight.data.copy_(pretrained_embedding)
#GRU.embedding.weight.data.copy_(pretrained_embedding)
print('embedding layer inited.')

optimizer = optim.Adam(LSTM.parameters(), lr=5e-4)
#optimizer = optim.Adam(GRU.parameters(), lr=5e-4)
torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.5, last_epoch=-1)
criteon = nn.BCEWithLogitsLoss().to(device)
LSTM.to(device)
#GRU.to(device)

train_loss = []
train_acc = []
test_loss = []
test_acc = []


def binary_acc(preds, y):
    """
    get accuracy
    """
    preds = torch.round(torch.sigmoid(preds))
    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criteon):
    avg_acc = []
    avg_loss = []
    model.train()

    for i, batch in enumerate(iterator):

        # [seq, b] => [b, 1] => [b]
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        pred = model(batch.text).squeeze(1)
        #print(pred.size())
        #print(batch.label.size())
        #
        loss = criteon(pred, batch.label)
        avg_loss.append(loss.item())
        acc = binary_acc(pred, batch.label).item()
        avg_acc.append(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(i, acc)

    avg_acc = np.array(avg_acc).mean()
    train_loss.append(np.array(avg_loss).mean())
    train_acc.append(avg_acc)
    print('train loss', np.array(avg_loss).mean())
    print('avg acc:', avg_acc)
    torch.save(model.state_dict(),'net_params.pth')


def eval(model, iterator, criteon):
    avg_acc = []
    avg_loss = []

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            # [b, 1] => [b]
            pred = model(batch.text).squeeze(1)

            #
            loss = criteon(pred, batch.label)
            avg_loss.append(loss.item())

            acc = binary_acc(pred, batch.label).item()
            avg_acc.append(acc)

    avg_acc = np.array(avg_acc).mean()
    test_loss.append(np.array(avg_loss).mean())
    test_acc.append(avg_acc)
    print('test loss', np.array(avg_loss).mean())
    print('>>test:', avg_acc)


for epoch in range(10):
    train(LSTM, train_iterator, optimizer, criteon)
    eval(LSTM, test_iterator, criteon)