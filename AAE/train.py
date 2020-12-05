from utils import to_var, loss_function
from model import Encoder, Decoder, Discriminator

def train(args, EPS, epoch, Enc, Dec, Discrim, optimizer, train_loader, device, log_interval):
    train_iter = iter(train_loader)

	# Fetch the images and labels and convert them to variables
	images, labels = next(train_iter)
	images, labels = to_var(images.view(images.size(0), -1)), to_var(labels)

    Enc.zero_grad()
    Dec.zero_grad()
    train_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))