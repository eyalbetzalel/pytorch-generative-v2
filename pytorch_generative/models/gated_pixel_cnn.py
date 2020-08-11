"""Implementation of the Gated PixelCNN [1].

TODO(eugenhotaj): describe.

We follow the implementaiton in [2] but use a casually masked GatedPixelCNNLayer
for the input instead of a causally masked Conv2d layer. For efficiency, the 
masked Nx1 and 1xN convolutions are implemented via unmasked (N//2+1)x1 and
1x(N//2+1) convolutions with padding and cropping, as suggested in [1].

NOTE: Our implementaiton does *not* use autoregressive channel masking. This
means that each output depends on whole pixels and not sub-pixels. For outputs
with multiple channels, other methods can be used, e.g. [3]. 

[1]: https://arxiv.org/abs/1606.05328
[2]: http://www.scottreed.info/files/iclr2017.pdf
[3]: https://arxiv.org/abs/1701.05517
"""

import torch
from torch import distributions
from torch import nn


class GatedActivation(nn.Module):
  """A function which computes 'tanh(f) * sigmoid(g)'.
  
  Here 'f' and 'g' correspond to the top 1/2 and bottom 1/2 of the channels of
  some input image.
  """

  def forward(self, x):
    _, c, _, _ = x.shape
    assert c % 2 == 0, 'x must have an even number of channels.'
    tanh, sigmoid = x[:, :c//2, :, :], x[:, c//2:, :, :]
    return torch.tanh(tanh) * torch.sigmoid(sigmoid)


class GatedPixelCNNLayer(nn.Module):
  """A Gated PixelCNN layer.

  N.B.: This layer does *not* implement autoregressive channel masking.

  The layer takes as input 'vstack' and 'hstack' from previous 
  'GatedPixelCNNLayers' and returns 'vstack', 'hstack', 'skip' where 'skip' is
  the skip connection to the pre-logits layer.
  """

  def __init__(self, in_channels, out_channels, kernel_size=3, is_causal=False):
    """Initializes a new GatedPixelCNNLayer instance.

    Args:
      in_channels: The number of channels in the input.
      out_channels: The number of output channels.
      kernel_size: The size of the (masked) convolutional kernel to use.
      is_causal: Whether the 'GatedPixelCNNLayer' is causal. If 'True', the 
        current pixel is masked out so the computation only depends on pixels
        to the left and above. The residual connection in the horizontal stack
        is also removed.
    """
    super().__init__()

    assert kernel_size % 2 == 1, 'kernel_size cannot be even'

    self._in_channels = in_channels
    self._out_channels = out_channels
    self._activation = GatedActivation()
    self._kernel_size = kernel_size
    self._padding = (kernel_size - 1) // 2  # (kernel_size - stride) / 2
    self._is_causal = is_causal

    # Vertical stack convolutions.
    self._vstack_1xN = nn.Conv2d(
        in_channels=self._in_channels, out_channels=self._out_channels, 
        kernel_size=(1, self._kernel_size),
        padding=(0, self._padding))
    # TODO(eugenhotaj): Is it better to shift down the the vstack_Nx1 output
    # instead of adding extra padding to the convolution? When we add extra 
    # padding, the cropped output rows will no longer line up with the rows of 
    # the vstack_1x1 output.
    self._vstack_Nx1 = nn.Conv2d(
        in_channels=self._out_channels, out_channels=2*self._out_channels,
        kernel_size=(self._kernel_size//2 + 1, 1),
        padding=(self._padding + 1, 0))
    self._vstack_1x1 = nn.Conv2d(
        in_channels=in_channels, out_channels=2*out_channels, kernel_size=1)

    self._link = nn.Conv2d(
        in_channels=2*out_channels, out_channels=2*out_channels, kernel_size=1)

    # Horizontal stack convolutions.
    self._hstack_1xN = nn.Conv2d(
        in_channels=self._in_channels, out_channels=2*self._out_channels,
        kernel_size=(1, self._kernel_size//2 + 1),
        padding=(0, self._padding + int(self._is_causal)))
    self._hstack_residual = nn.Conv2d(
        in_channels=out_channels, out_channels=out_channels, kernel_size=1)
    self._hstack_skip = nn.Conv2d(
        in_channels=out_channels, out_channels=out_channels, kernel_size=1)

  def forward(self, vstack_input, hstack_input):
    """Computes the forward pass.
    
    Args:
      vstack_input: The input to the vertical stack.
      hstack_input: The input to the horizontal stack.
    Returns:
      (vstack,  hstack, skip) where vstack and hstack are the vertical stack
      and horizontal stack outputs respectively and skip is the skip connection
      output. 
    """
    _, _, h, w = vstack_input.shape  # Assuming NCHW.

    # Compute vertical stack.
    vstack = self._vstack_Nx1(self._vstack_1xN(vstack_input))[:, :, :h, :]
    link = self._link(vstack)
    vstack += self._vstack_1x1(vstack_input)
    vstack = self._activation(vstack)

    # Compute horizontal stack.
    hstack = link + self._hstack_1xN(hstack_input)[:, :, :, :w]
    hstack = self._activation(hstack)
    skip = self._hstack_skip(hstack)
    hstack = self._hstack_residual(hstack)
    # NOTE(eugenhotaj): We cannot use a residual connection for causal layers
    # otherwise we'll have access to future pixels.
    if not self._is_causal:
      hstack += hstack_input

    return vstack, hstack, skip


class GatedPixelCNN(nn.Module):
  """The Gated PixelCNN model."""

  def __init__(self, 
               in_channels, 
               out_dim=1,
               n_gated=10,
               gated_channels=128,
               head_channels=32):
    """Initializes a new GatedPixelCNN instance.
    
    Args:
      in_channels: The number of channels in the input.
      out_dim: The dimensionality of the output. Given input of the form
        (N, C, H, W), the output from the GatedPixelCNN model will be 
        (N, out_dim, C, H, W).
      n_gated: The number of gated layers (not including the input layers).
      gated_channels: The number of channels to use in the gated layers.
      head_channels: The number of channels to use in the 1x1 convolution blocks
        in the head after all the gated channels.
    """

    super().__init__()
    self._out_dim = out_dim
    self._input = GatedPixelCNNLayer(
      in_channels=in_channels,
      out_channels=gated_channels,
      kernel_size=7,
      is_causal=True)
    self._gated_layers = nn.ModuleList([
        GatedPixelCNNLayer(in_channels=gated_channels, 
                           out_channels=gated_channels,
                           kernel_size=3,
                           is_causal=False)
        for _ in range(n_gated)
    ])
    self._head = nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(in_channels=gated_channels, 
                  out_channels=head_channels, 
                  kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=head_channels, 
                  out_channels=self._out_dim * in_channels,
                  kernel_size=1),
        nn.Sigmoid())

  def forward(self, x):
    n, c, h, w = x.shape
    vstack, hstack, skip_connections = self._input(x, x)
    for gated_layer in self._gated_layers:
      vstack, hstack, skip = gated_layer(vstack, hstack)
      skip_connections += skip
    return self._head(skip_connections).view((n, self._out_dim, c, h, w))

  # TODO(eugenhotaj): We need to update the sampling code so it can handle 
  # outputs with dim > 1. One thing that's unclear: should the sample method
  # be part of the model?
  def sample(self, condition_on=None):
    """Samples a new image.
    
    Args:
      conditioned_on: An (optional) image to condition samples on. Only 
        dimensions with values < 0 will be sampled. For example, if 
        conditioned_on[i] = -1, then output[i] will be sampled conditioned on
        dimensions j < i. If 'None', an unconditional sample will be generated.
    """
    with torch.no_grad():
    
      if conditioned_on is None:
        device = next(self.parameters()).device
        conditioned_on = (torch.ones((1, self._input_dim)) * - 1).to(device)
      else:
        conditioned_on = conditioned_on.clone()

      for row in range(28):
        for column in range(28):
          for channel in range(1):
            out = self.forward(conditioned_on).squeeze(dim=1)[:, channel, row, column]
            out = distributions.Bernoulli(probs=out).sample()
            conditioned_on[:, channel, row, column] = torch.where(
                conditioned_on[:, channel, row, column] < 0, out, 
                conditioned_on[:, channel, row, column])
      return conditioned_on