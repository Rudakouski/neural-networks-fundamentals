import torch

class Autoencoder(torch.nn.Module):
    def __init__(
            self,
            channels,
            activation=torch.nn.ReLU):
        ...
        super().__init__()
        ## YOUR CODE HERE
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(channels[0], channels[1]),
            activation(),
            torch.nn.Linear(channels[1], channels[2])
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(channels[2], channels[1]),
            activation(),
            torch.nn.Linear(channels[1], channels[0])
        )
        if not hasattr(self, 'encoder'):
            self.encoder = torch.nn.Identity()
        if not hasattr(self, 'decoder'):
            self.decoder = torch.nn.Identity()

    def __forward_kernel(self, signal):
        input_shape = signal.shape
        res = signal
        ## YOUR CODE HERE
        flattened = res.view(res.size(0), -1)
        encoded = self.encoder(flattened)
        reconstructed = self.decoder(encoded)
        res = reconstructed
        res = res.reshape(input_shape)
        return res

    def forward(self, batch):
        ## YOUR CODE HERE
        image = batch['data']['image']
        reconstruction = self.__forward_kernel(image)
        if 'signals' not in batch:
            batch['signals'] = {}
        batch['signals']['reconstruction'] = reconstruction
        if 'signals' not in batch:
            batch['signals'] = {'reconstruction': batch['data']['image']}
        return batch