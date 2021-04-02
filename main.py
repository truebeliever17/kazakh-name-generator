import torch
import torch.nn.functional as F
import pickle
import numpy as np
from model import LSTMGenerator


def encode_to_one_hot(x, vocab_size):
    x_encoded = torch.zeros((1, 1, vocab_size))
    x_encoded[:, :, x] = 1

    return x_encoded


def load_char2idx(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_model(path, device, vocab_size, hidden_dim, dropout):
    model = LSTMGenerator(vocab_size, hidden_dim, dropout)
    model.load_state_dict(torch.load(path))
    model.to(device)
    return model


def generate_names(model, device, char2idx, idx2char, vocab_size, first_letter='SOS', n=10, max_length=50):
    generated_names = []

    for _ in range(n):
        name = first_letter if first_letter != 'SOS' else ''
        current_char = encode_to_one_hot(char2idx[first_letter], vocab_size).to(device)
        prev_state = model.init_state(device)

        model.eval()
        with torch.no_grad():
            for _ in range(max_length):
                prediction, prev_state = model(current_char, prev_state)

                p = F.softmax(prediction, dim=1).detach().cpu().numpy()
                idx = np.random.choice(vocab_size, p=p.squeeze())
                next_char = idx2char[idx]

                if next_char == 'EOS':
                    break

                name += next_char
                current_char = encode_to_one_hot(idx, vocab_size).to(device)

        generated_names.append(name)

    return generated_names


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    char2idx = load_char2idx('char2idx.pkl')
    idx2char = {v: k for k, v in char2idx.items()}

    vocab_size = len(char2idx)
    hidden_dim = 256
    dropout = 0.2

    model = load_model('model.pt', device, vocab_size, hidden_dim, dropout)

    for name in generate_names(model, device, char2idx, idx2char, vocab_size):
        print(name)
