# Exporting a Model to ONNX

[ONNX](https://onnx.ai/) is a standard format for representing machine learning models. Sample Factory can export models to ONNX format.

Exporting to ONNX allows you to:

- Deploy your model in various production environments
- Use hardware-specific optimizations provided by ONNX Runtime
- Integrate your model with other tools and frameworks that support ONNX

## Usage Examples

First, train a model using Sample Factory.

```bash
python -m sf_examples.train_gym_env --experiment=example_gym_cartpole-v1 --env=CartPole-v1 --use_rnn=False --reward_scale=0.1
```

Then, use the following command to export it to ONNX:

```bash
python -m sf_examples.export_onnx_gym_env --experiment=example_gym_cartpole-v1 --env=CartPole-v1 --use_rnn=False
```

This creates `example_gym_cartpole-v1.onnx` in the current directory.

### Using the Exported Model

Here's how to use the exported ONNX model:

```python
import numpy as np
import onnxruntime

ort_session = onnxruntime.InferenceSession("example_gym_cartpole-v1.onnx", providers=["CPUExecutionProvider"])

# The model expects a batch of observations as input.
batch_size = 3
ort_inputs = {"obs": np.random.rand(batch_size, 4).astype(np.float32)}

ort_out = ort_session.run(None, ort_inputs)

# The output is a list of actions, one for each observation in the batch.
selected_actions = ort_out[0]
print(selected_actions) # e.g. [1, 1, 0]
```

### RNN

When exporting a model that uses RNN with `--use_rnn=True` (default), the model will expect RNN states as input.
Note that for RNN models, the batch size must be 1.

```python
import numpy as np
import onnxruntime

ort_session = onnxruntime.InferenceSession("rnn.onnx", providers=["CPUExecutionProvider"])

rnn_states_input = next(input for input in ort_session.get_inputs() if input.name == "rnn_states")
rnn_states = np.zeros(rnn_states_input.shape, dtype=np.float32)
batch_size = 1 # must be 1

for _ in range(10):
  ort_inputs = {"obs": np.random.rand(batch_size, 4).astype(np.float32), "rnn_states": rnn_states}
  ort_out = ort_session.run(None, ort_inputs)
  rnn_states = ort_out[1] # The second output is the updated rnn states
```

## Configuration

The following key parameters will change the behavior of the exported mode:

- `--use_rnn` Whether the model uses RNN. See the RNN example above.

- `--eval_deterministic` If `True`, actions are selected by argmax.
