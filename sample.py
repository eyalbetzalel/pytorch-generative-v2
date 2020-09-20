def _get_conditioned_on(out_shape, conditioned_on, device):
    assert out_shape is None or conditioned_on is None, \
      'Must provided one, and only one of "out_shape" or "conditioned_on"'
    if conditioned_on is None:
      conditioned_on = (torch.ones(out_shape) * - 1).to(device)
    else:
      conditioned_on = conditioned_on.clone()
    return conditioned_on

def sample(model, out_shape=None, conditioned_on=None):
  """Generates new samples from the model.
  The model output is assumed to be the parameters of either a Bernoulli or 
  multinoulli (Categorical) distribution depending on its dimension.
  Args:
    out_shape: The expected shape of the sampled output in NCHW format. 
      Should only be provided when 'conditioned_on=None'.
    conditioned_on: A batch of partial samples to condition the generation on.
      Only dimensions with values < 0 will be sampled while dimensions with 
      values >= 0 will be left unchanged. If 'None', an unconditional sample
      will be generated.
  """
  device = next(model.parameters()).device
  with torch.no_grad():
    conditioned_on = _get_conditioned_on(out_shape, conditioned_on, device)
    n, c, h, w = conditioned_on.shape
    for row in range(h):
      for column in range(w):
        out = model.forward(conditioned_on)[:, :, row, column]
        distribution = (distributions.Categorical if out.shape[1] > 1 
                        else distributions.Bernoulli)
        out = distribution(probs=out).sample()
        conditioned_on[:, :, row, column] = torch.where(
            conditioned_on[:, :, row, column] < 0,
            out, 
            conditioned_on[:, :, row, column])
    return conditioned_on
