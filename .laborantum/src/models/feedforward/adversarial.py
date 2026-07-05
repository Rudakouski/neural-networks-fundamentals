import torch
import copy


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, signal, strength):
        ctx.strength = strength
        return signal.view_as(signal)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.strength * grad_output, None


class GradientReversalLayer(torch.nn.Module):
    def __init__(self, strength=1.0):
        super().__init__()
        self.strength = float(strength)

    def forward(self, signal):
        return GradientReversalFunction.apply(signal, self.strength)


class GAN(torch.nn.Module):
    def __init__(
        self,
        channels,
        gradient_reversal_strength=1.0,
        activation=lambda: torch.nn.LeakyReLU(negative_slope=0.5)
    ):
        super().__init__()
        self.generator_discriminator_bridge = GradientReversalLayer(gradient_reversal_strength)
        self.gradient_reversal = self.generator_discriminator_bridge

        gen_layers = []
        for i in range(len(channels) - 1):
            gen_layers.append(torch.nn.Linear(channels[i], channels[i+1]))
            if i < len(channels) - 2:          # после скрытых слоёв, но не после последнего
                gen_layers.append(activation())
        gen_layers.append(torch.nn.Tanh())
        self.generator = torch.nn.Sequential(*gen_layers)

        rev_channels = list(reversed(channels))
        disc_layers = []
        for i in range(len(rev_channels) - 1):
            disc_layers.append(torch.nn.Linear(rev_channels[i], rev_channels[i+1]))
            if i < len(rev_channels) - 2:
                disc_layers.append(activation())
        self.discriminator = torch.nn.Sequential(*disc_layers)

        self.classifier = torch.nn.Linear(rev_channels[-1], 1)

    def discriminate(self, signal):
        signal = signal.reshape(signal.shape[0], -1)
        features = self.discriminator(signal)
        return self.classifier(features).flatten()

    def forward(self, batch):
        noise = batch['data'].get('noise')
        if noise is None:
            device = batch['data'].get('image', torch.empty(0)).device
            noise = torch.empty(0, device=device)
        B = noise.shape[0]
        device = noise.device

        generated = self.generator(noise)

        reversed_generated = self.gradient_reversal(generated)

        real_images = batch['data'].get('image') or batch['data'].get('real')
        if real_images is not None:
            real_flatten = real_images.reshape(real_images.shape[0], -1)
            combined = torch.cat([reversed_generated, real_flatten], dim=0)
        else:
            combined = reversed_generated

        disc_features = self.discriminator(combined)
        logits = self.classifier(disc_features).flatten()

        if real_images is not None:
            fake_logits = logits[:B]
            real_logits = logits[B:]
        else:
            fake_logits = logits
            real_logits = None

        signals = {
            'generated': generated,
            'discriminator_logits': logits,
            'fake_logits': fake_logits,
            'discriminator_scores': logits,
            'fake_scores': fake_logits,
        }
        if real_logits is not None:
            signals['real_logits'] = real_logits
            signals['real_scores'] = real_logits
        batch['signals'] = signals

        postprocessed = {
            'discriminator_score': logits,
            'fake_score': fake_logits,
            'discriminator_probability': torch.sigmoid(logits),
            'fake_probability': torch.sigmoid(fake_logits),
        }
        if real_logits is not None:
            postprocessed['real_score'] = real_logits
            postprocessed['real_probability'] = torch.sigmoid(real_logits)
        batch['postprocessed'] = postprocessed

        if 'signals' not in batch:
            generated = batch['data'].get('noise')
            if generated is None:
                generated = torch.empty(0)
            batch['signals'] = {
                'generated': generated,
                'fake_scores': torch.zeros(generated.shape[0], device=generated.device),
                'fake_logits': torch.zeros(generated.shape[0], device=generated.device),
            }
            batch['postprocessed'] = {
                'fake_score': torch.zeros(generated.shape[0], device=generated.device),
                'fake_probability': torch.zeros(generated.shape[0], device=generated.device),
            }
        return batch