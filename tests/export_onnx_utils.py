import onnx
import onnxruntime
import torch

from sample_factory.algo.utils.make_env import BatchedVecEnv
from sample_factory.enjoy import make_env
from sample_factory.export_onnx import OnnxExporter, create_onnx_exporter, export_onnx, generate_args
from sample_factory.utils.typing import Config
from sample_factory.utils.utils import experiment_dir


def to_numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def check_onnx_model(filename: str) -> None:
    model = onnx.load(filename)
    onnx.checker.check_model(model)


def check_rnn_inference_result(
    env: BatchedVecEnv, model: OnnxExporter, ort_session: onnxruntime.InferenceSession
) -> None:
    rnn_states_input = next(input for input in ort_session.get_inputs() if input.name == "rnn_states")
    rnn_states = torch.zeros(rnn_states_input.shape, dtype=torch.float32)
    ort_rnn_states = to_numpy(rnn_states)

    for _ in range(3):
        args = generate_args(env.observation_space)
        actions, rnn_states = model(**args, rnn_states=rnn_states)

        ort_inputs = {k: to_numpy(v) for k, v in args.items()}
        ort_inputs["rnn_states"] = ort_rnn_states
        ort_out = ort_session.run(None, ort_inputs)
        ort_rnn_states = ort_out[1]

        assert (to_numpy(actions) == ort_out[0]).all()


def check_inference_result(env: BatchedVecEnv, model: OnnxExporter, ort_session: onnxruntime.InferenceSession) -> None:
    for batch_size in [1, 3]:
        args = generate_args(env.observation_space, batch_size)
        actions = model(**args)

        ort_inputs = {k: to_numpy(v) for k, v in args.items()}
        ort_out = ort_session.run(None, ort_inputs)

        assert len(ort_out[0]) == batch_size
        assert (to_numpy(actions) == ort_out[0]).all()


def check_export_onnx(cfg: Config) -> None:
    cfg.eval_deterministic = True
    directory = experiment_dir(cfg=cfg, mkdir=False)
    filename = f"{directory}/{cfg.experiment}.onnx"
    status = export_onnx(cfg, filename)
    assert status == 0

    check_onnx_model(filename)

    env = make_env(cfg)
    model = create_onnx_exporter(cfg, env, enable_jit=True)
    ort_session = onnxruntime.InferenceSession(filename, providers=["CPUExecutionProvider"])

    if cfg.use_rnn:
        check_rnn_inference_result(env, model, ort_session)
    else:
        check_inference_result(env, model, ort_session)
