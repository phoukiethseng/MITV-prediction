import torch
from torch import optim

from dataset import MITVDataset
from model import MITVLSTM
from torch.nn.functional import mse_loss

import torch
from torch.utils.tensorboard import SummaryWriter

NUM_WORKERS = 0
EPOCHS = 100
BATCH_SIZE = 128

def collate_fn(batch):
    return batch

def main():
    device = 'cpu'
    if torch.cuda.is_available():
        torch.set_default_device('cuda')
        device = 'cuda'

    torch.set_default_dtype(torch.float32)

    dataset = MITVDataset()
    train, test = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=torch.Generator(device=device))

    train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, generator=torch.Generator(device=device), collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, generator=torch.Generator(device=device), collate_fn=collate_fn)

    # Our LSTM model
    model = MITVLSTM(input_size=60, hidden_size=256)

    # Using SGD optimizer
    SGD = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

    # Learning Rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(SGD, gamma=0.8)

    model.train()

    writer = SummaryWriter()
    for epoch in range(EPOCHS):
        train_loss = train_loop(model, train_loader, optimizer=SGD, device=device)
        test_loss = test_loop(model, test_loader, device=device)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        lr_scheduler.step()

    return None

def train_loop(model, dataloader, optimizer, device='cpu'):
    num_batches = len(dataloader.dataset) // BATCH_SIZE
    accum_loss = 0
    for batch, data in enumerate(dataloader):
        X, Y, seq_len = zip(*data)
        optimizer.zero_grad()
        pred = model(X, seq_len)

        Y = torch.nn.utils.rnn.pad_sequence(Y, batch_first=True, padding_value=0)
        loss = mse_loss(pred.squeeze(), Y, reduction='mean')
        accum_loss += loss.item()
        # print(f"batch #{batch+1}: loss {loss.item()}")
        loss.backward()
        optimizer.step()

    return accum_loss / num_batches


def test_loop(model, dataloader, device='cpu'):
    num_batches = len(dataloader.dataset) // BATCH_SIZE
    accum_loss = 0
    model.eval()
    for batch, data in enumerate(dataloader):
        X, Y, seq_len = zip(*data)
        pred = model(X, seq_len)

        Y = torch.nn.utils.rnn.pad_sequence(Y, batch_first=True, padding_value=0)
        loss = mse_loss(pred.squeeze(), Y, reduction='mean')
        accum_loss += loss.item()
    model.train()
    return accum_loss / num_batches

if __name__ == '__main__':
    main()