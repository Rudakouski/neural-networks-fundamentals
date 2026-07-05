import torch
import torch.nn as nn

class SimpleFCNN(torch.nn.Module):
    def __init__(
            self, 
            channels=None,
            n_classes=10,
            activation=torch.nn.ReLU):
        ...
        ## YOUR CODE HERE
        # Define network modules in the constructor
        super().__init__()
        
        if channels is None:
            channels = [128, 64]
        
        layers = []
        in_features = 28 * 28
        for out_features in channels:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(activation())
            in_features = out_features
        layers.append(nn.Linear(in_features, n_classes))
        
        self.net = nn.Sequential(*layers) 

        
    def __forward_kernel(self, signal):
        signal = signal.reshape([signal.shape[0], -1])
        ## YOUR CODE HERE
        # Pass the signal through the modules in forward
        signal = self.net(signal)
        return signal

    def forward(self, batch):
        signal = batch['data']['image']
        signal = self.__forward_kernel(signal)
        
        # Put the result into the batch
        batch['signals'] = {'output': signal}
        
        # Perform postprocessing after we get the output
        self.postprocessing(batch)
        
        return batch['signals']['output']
    
#    def postprocessing(self, batch):
#        
#        # Take network's output from the batch
#        signal = batch['signals']['output']
#        
#        ## YOUR CODE HERE
#        predicted_classes = torch.argmax(signal, dim=1)
#        # Put the processed result into the batch
#        batch['postprocessed'] = {'class': signal}
    def postprocessing(self, batch):
        signal = batch['signals']['output']
        predicted_classes = torch.argmax(signal, dim=1)
        batch['postprocessed'] = {'class': predicted_classes}
