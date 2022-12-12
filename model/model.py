# importing modules
import torch.nn as nn
import torch.tensor as tensor
from torch import no_grad
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# defining the required parameters
class DatasetClass(torch.utils.data.Dataset):
    def __init__(self, filepath):
        self.data = pd.read_csv(filepath, header=0)
        
    def __getitem__(self, idx):
        return self.data.y[idx], self.data.x[idx]
    
    def __len__(self):
        return len(self.data)

dataset = DatasetClass("intent_pair.csv")

tokenizer = get_tokenizer('basic_english')
train_iter = iter(dataset)

def yield_tokens(data_iter):
    for idx in range(data_iter.__len__()):
        yield tokenizer(dataset[idx][1])

vocab = build_vocab_from_iterator(yield_tokens(dataset), specials=["<unk>"])
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x)

# the model class
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

model = TextClassificationModel()
model.load_state_dict('weights.pth')

label_list = {
    0: "hello_intent",
    1: "whoami_intent",
    2: "question_intent",
    3: "bye_intent",
}

def predict(text):
    with no_grad():
        text = tensor(text_pipeline(text))
        output = model(text, tensor([0]))
        return output.argmax(1).item()
