import torch


class ResidualBottleneck(torch.nn.Module):
    def __init__(
            self, 
            in_channels,
            out_channels,
            prenormalization=lambda n_channels: torch.nn.Identity(),
            postnormalization=lambda n_channels: torch.nn.Identity(),
            activation=torch.nn.ReLU,
            compression=1,
            residual=True):

        super().__init__()
        self.block = torch.nn.Identity()
        self.bypass = torch.nn.Identity()
        self.residual = residual

        hidden_channels = max(1, in_channels // compression)

        self.prenorm = prenormalization(in_channels)
        self.postnorm = postnormalization(out_channels)
        self.activation = activation()

        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            self.activation,
            torch.nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

        if residual and in_channels != out_channels:
            self.bypass = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.bypass = torch.nn.Identity()

    def forward(self, signal):
        ## YOUR CODE HERE
        x = self.prenorm(signal)

        out = self.block(x)

        if self.residual:
            out = out + self.bypass(x)

        out = self.postnorm(out)

        signal = out
        return signal


class FullyConvolutionalNN(torch.nn.Module):
    def __init__(
            self,
            block=lambda in_channels, out_channels: torch.nn.Conv2d(in_channels, out_channels, (1, 1)),
            in_channels=1,
            mid_channels=[16, 32, 64, 128],
            out_channels=10,
            n_blocks=[1, 1, 1, 1]):
        ...
        ## YOUR CODE HERE
        super().__init__()
        ## YOUR CODE HERE

        dims = [28 * 28] + mid_channels + [out_channels]

        layers = []
        for i in range(len(dims) - 1):
            layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(torch.nn.ReLU())

        self.model = torch.nn.Sequential(*layers)


    def __forward_kernel(self, signal):
        ## YOUR CODE HERE
        # Pass the signal through the modules in forward

        signal = signal.view(signal.size(0), -1)  # [B, 784]
        signal = self.model(signal)
        return signal

    def forward(self, batch):
        signal = batch['data']['image']
        signal = self.__forward_kernel(signal)

        # Put the result into the batch
        batch['signals'] = {'output': signal}

        # Perform postprocessing after we get the output
        self.postprocessing(batch)

        return batch

    def postprocessing(self, batch):

        # Take network's output from the batch
        signal = batch['signals']['output']

        signal = torch.argmax(signal, dim=1)

        # Put the processed result into the batch
        batch['postprocessed'] = {'class': signal}
