# from asyncio.sslproto import add_flowcontrol_defaults
from email import header
from logging import warning
import torch
from torch import Tensor, device, nn

import torch.nn.functional as F


from sample_factory.model.model_utils import model_device
from sample_factory.utils.typing import Config, ObsSpace
from sample_factory.utils.utils import log


from sample_factory.model.core import ModelCore,ModelCoreIdentity,ModelCoreRNN


from sample_factory.model.encoder import Encoder, make_img_encoder, DepthEncoder
from sf_examples.dmlab.dmlab30 import DMLAB_INSTRUCTIONS, DMLAB_VOCABULARY_SIZE

from sf_examples.dmlab.dmlab_model import DmlabEncoder



class DGProjectionBatchNovelty(nn.Module):
    def __init__(self, feature_dim, pattern_limit, detection_threshold=0.1, novelty_threshold=0.4, 
                 eps=1e-8, norm_coef=0.002, soft_gate_scale=10.0, bias=3.0):
        """
        Args:
          feature_dim (int): Dimension of the input features.
          pattern_limit (int): Fixed number of output units (stored patterns).
          detection_threshold (float): Threshold for activation based on the projection magnitude.
          novelty_threshold (float): Threshold for novelty detection based on singular values.
          eps (float): Small constant to avoid division by zero.
          norm_coef (float): Coefficient used in pattern replacement.
          soft_gate_scale (float): Scaling factor for the sigmoid.
          bias (float): Bias subtracted in the sigmoid so that at the detection threshold, the output is near zero.
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.pattern_limit = pattern_limit
        self.detection_threshold = detection_threshold
        self.novelty_threshold = novelty_threshold
        self.eps = eps
        self.norm_coef = norm_coef
        self.soft_gate_scale = soft_gate_scale
        self.bias = bias

        # Fixed buffer for stored patterns: shape (pattern_limit, feature_dim).
        self.register_buffer('patterns', torch.zeros(pattern_limit, feature_dim))
        # Boolean mask indicating which slots are active.
        self.register_buffer('pattern_mask', torch.zeros(pattern_limit, dtype=torch.bool))
        # Tensors to track cumulative activation counts and sample counts per slot.
        self.register_buffer('pattern_activation_counts', torch.zeros(pattern_limit))
        self.register_buffer('pattern_sample_counts', torch.zeros(pattern_limit))

    def forward(self, x):
        # Normalize input samples (differentiable operations).
        x_norm = x / (x.norm(dim=1, keepdim=True) + self.eps)
        batch_size = x_norm.size(0)

        # Update sample counts for active slots (buffer update outside of autograd).
        if self.pattern_mask.any():
            with torch.no_grad():
                self.pattern_sample_counts[self.pattern_mask] += batch_size

        # --- Novelty Detection ---
        if self.pattern_mask.any():
            active_patterns = self.patterns[self.pattern_mask]  # (n_active, feature_dim)
            proj = torch.matmul(x_norm, active_patterns.t())      # (batch_size, n_active)
            proj_span = torch.matmul(proj, active_patterns)         # (batch_size, feature_dim)
            x_null = x_norm - proj_span
        else:
            x_null = x_norm

        try:
            U, S, Vh = torch.linalg.svd(x_null, full_matrices=False)
            novel_mask = S > self.novelty_threshold
            k = int(novel_mask.sum().item())
            if k > 0:
                new_patterns = Vh[:k, :]
                new_patterns = F.normalize(new_patterns, p=2, dim=1)
                new_norms = S[:k] / batch_size
                with torch.no_grad():
                    self.update_patterns(new_patterns, new_norms, batch_size)
        except Exception as e:
            print("SVD failed:", e)

        # --- Differentiable Activation Computation ---
        # Compute similarity between normalized inputs and stored patterns.
        sim = torch.matmul(x_norm, self.patterns.t())  # (batch_size, pattern_limit)
        # Zero out similarities for inactive pattern slots.
        sim = sim * self.pattern_mask.to(sim.dtype)

        # Compute a differentiable activation for each pattern slot.
        # For each element, when sim is equal to detection_threshold,
        # the input to sigmoid becomes: soft_gate_scale*(0) - bias = -bias.
        # With bias=3.0, sigmoid(-3) is approximately 0.047, so near-zero.
        activations = torch.sigmoid(self.soft_gate_scale * (sim - self.detection_threshold) - self.bias)
        
        # Bookkeeping: update activation counts using a hard decision for pattern with highest similarity.
        with torch.no_grad():
            # We still identify a "most activated" pattern per sample for bookkeeping.
            _, max_indices = torch.max(sim, dim=1)
            # Use a simple threshold on the computed activations to decide if a sample is active.
            active_samples = (activations.max(dim=1)[0] > self.detection_threshold)
            if active_samples.any():
                upd = torch.bincount(max_indices[active_samples], minlength=self.pattern_limit).float()
                self.pattern_activation_counts += upd

        return activations

    def update_patterns(self, new_patterns, new_norms, current_batch_size):
        k_new = new_patterns.size(0)
        device = self.patterns.device

        # Compute normalized activation counts for stored patterns.
        stored_norm = self.pattern_activation_counts / (self.pattern_sample_counts + self.eps)
        stored_norm = stored_norm.to(device)

        stored_idx = torch.arange(self.pattern_limit, device=device)
        sorted_stored_idx = stored_idx[torch.argsort(stored_norm)]
        
        new_idx = torch.arange(k_new, device=new_patterns.device)
        sorted_new_idx = new_idx[torch.argsort(new_norms, descending=True)]
        
        replace_count = 0
        for new_i in sorted_new_idx:
            if replace_count < self.pattern_limit:
                candidate_stored_idx = sorted_stored_idx[replace_count]
                if new_norms[new_i] * self.norm_coef > stored_norm[candidate_stored_idx]:
                    self.patterns[candidate_stored_idx] = new_patterns[new_i]
                    self.pattern_activation_counts[candidate_stored_idx] = 0.0
                    self.pattern_sample_counts[candidate_stored_idx] = current_batch_size
                    self.pattern_mask[candidate_stored_idx] = True
                    replace_count += 1
        log.info(f"{replace_count} patterns replaced, batch size: {current_batch_size}")





class DGProjection(nn.Module):
    def __init__(self, in_features: int, out_features: int, dg_lr: float = 0.001, weight_decay: float = 0.01):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dg_lr = dg_lr
        self.weight_decay = weight_decay

        self.linear = nn.Linear(self.in_features, out_features, bias=False)
        self.activation = nn.ReLU()

        # Disable gradients for the linear layer as updates are custom.
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute raw output and apply activation.
        raw_output = self.linear(x)
        raw_output = self.activation(raw_output)

        # Create a "corrected" output: use a baseline of 0.1 for non-winning neurons,
        # and set the winning neuron to 1.0.
        with torch.no_grad():
            corrected_homeo = torch.full_like(raw_output, 0.1)
            max_idx = torch.argmax(raw_output, dim=1, keepdim=True)
            corrected_homeo.scatter_(1, max_idx, 1.0)

            corrected = torch.full_like(raw_output, 0)
            max_idx = torch.argmax(raw_output, dim=1, keepdim=True)
            corrected.scatter_(1, max_idx, 1.0)

        # Calculate the difference between the raw output and the corrected target.
        diff = raw_output - corrected_homeo

        # Compute the average weight update over the batch.
        batch_size = x.size(0)
        dW = torch.matmul(diff.t(), x) / batch_size

        # Update the weight matrix using the custom rule with weight decay.
        with torch.no_grad():
            # Incorporate weight decay directly into the update:
            self.linear.weight -= self.dg_lr * (dW + self.weight_decay * self.linear.weight)

        # Return the corrected output as the projection result.
        return corrected

class DGProjection_obsolete(nn.Module):
    def __init__(self, in_features: int, out_features: int, dg_lr: float = 0.001, weight_decay: float = 0.01):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dg_lr = dg_lr
        self.weight_decay = weight_decay

        self.linear = nn.Linear(self.in_features, out_features, bias=False)
        self.activation = nn.ReLU()

        # Disable gradients for the linear layer as updates are custom.
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute raw output and apply activation.
        raw_output = self.linear(x)
        raw_output = self.activation(raw_output)

        # Create a "corrected" output: use a baseline of 0.1 for non-winning neurons,
        # and set the winning neuron to 1.0.
        with torch.no_grad():
            corrected_homeo = torch.full_like(raw_output, 0.1)
            max_idx = torch.argmax(raw_output, dim=1, keepdim=True)
            corrected_homeo.scatter_(1, max_idx, 1.0)

            corrected = torch.full_like(raw_output, 0)
            max_idx = torch.argmax(raw_output, dim=1, keepdim=True)
            corrected.scatter_(1, max_idx, 1.0)

        # Calculate the difference between the raw output and the corrected target.
        diff = raw_output - corrected_homeo

        # Compute the average weight update over the batch.
        batch_size = x.size(0)
        dW = torch.matmul(diff.t(), x) / batch_size

        # Update the weight matrix using the custom rule with weight decay.
        with torch.no_grad():
            # Incorporate weight decay directly into the update:
            self.linear.weight -= self.dg_lr * (dW + self.weight_decay * self.linear.weight)

        # Return the corrected output as the projection result.
        return corrected

class DGProjection_relu(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        """
        Enforces that each neuron's output (after softmax) is activated (set to 1)
        only if its probability exceeds the running quantile (e.g., 98th percentile)
        across previous batches.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.activation= nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)  # Shape: [batch_size, out_features]
        # Replace logits with softmaxed probabilities.
        probs = self.activation(logits)

        return probs
    
class DGProjection_batchnorm_relu(nn.Module):
    def __init__(self, in_features: int, out_features: int, intercept = 2):
        """
        Enforces that each neuron's output (after softmax) is activated (set to 1)
        only if its probability exceeds the running quantile (e.g., 98th percentile)
        across previous batches.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.batchnorm1d = nn.BatchNorm1d(out_features, affine=False,momentum=0.05)
        self.activation= nn.ReLU()
        self.intercept=intercept

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)  # Shape: [batch_size, out_features]
        x = self.batchnorm1d(x)
        # Replace logits with softmaxed probabilities.
        x = self.activation(x - self.intercept)

        return x

class DGProjection_log_softmax(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        """
        Enforces that each neuron's output (after softmax) is activated (set to 1)
        only if its probability exceeds the running quantile (e.g., 98th percentile)
        across previous batches.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.log_softmax= nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)  # Shape: [batch_size, out_features]
        # Replace logits with softmaxed probabilities.
        probs = self.log_softmax(logits)

        return probs

class DGProjection_simple_softmax(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        """
        Enforces that each neuron's output (after softmax) is activated (set to 1)
        only if its probability exceeds the running quantile (e.g., 98th percentile)
        across previous batches.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)  # Shape: [batch_size, out_features]
        # Replace logits with softmaxed probabilities.
        probs = F.softmax(logits, dim=1)

        return probs


class DGProjectionBatchSparsity(nn.Module):
    def __init__(self, in_features: int, out_features: int, active_percentage: float = 0.05):
        """
        Args:
            in_features: Number of input features.
            out_features: Number of output neurons.
            active_percentage: Desired percentage (per output neuron) of active samples in a batch.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.active_percentage = active_percentage
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        For each output neuron, keep only the top-k activations across the batch,
        where k is calculated to be about active_percentage of the batch size.
        """
        logits = self.linear(x)  # shape: [batch_size, out_features]
        batch_size = logits.size(0)
        # Determine number of samples to activate per neuron (at least 1)
        k = max(1, int(self.active_percentage * batch_size))
        
        # Transpose to shape [out_features, batch_size] so we can work per neuron
        logits_t = logits.transpose(0, 1)  # shape: [out_features, batch_size]
        
        # For each output neuron, find the indices of the top-k samples
        _, indices = torch.topk(logits_t, k, dim=1)
        
        # Create a zero mask of the same shape as logits_t
        mask = torch.zeros_like(logits_t)
        # Scatter ones into the top-k positions for each neuron
        mask.scatter_(1, indices, 1.0)
        
        # Transpose the mask back to [batch_size, out_features]
        mask = mask.transpose(0, 1)
        
        # Use a straight-through estimator trick:
        # In the forward pass, the output is the hard mask (exactly 1% active),
        # while in the backward pass, gradients flow as if the operation were the identity.
        output = mask + logits - logits.detach()
        return output





class DGProjection_simple_top1(nn.Module):
    def __init__(self, in_features: int, out_features: int, temperature: float = 1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.temperature = temperature

        # Standard linear layer with bias.
        self.linear = nn.Linear(self.in_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute logits from the linear layer.
        logits = self.linear(x)
        
        # Use Gumbel Softmax with the 'hard' flag set to True.
        # This returns one-hot vectors during the forward pass, but maintains gradients via the soft approximation.
        output = F.gumbel_softmax(logits, tau=self.temperature, hard=True)
        return output
    

class RunningActivationQuantile(nn.Module):
    def __init__(self, out_features, momentum=0.2, quantile=0.98):
        """
        Maintains a running estimate of the quantile (e.g., 98th percentile) for each neuron.
        
        Args:
            out_features (int): Number of output neurons.
            momentum (float): Weighting factor for new data.
            quantile (float): The desired quantile (e.g., 0.98 for the 98th percentile).
        """
        super().__init__()
        self.register_buffer("running_quantile", torch.zeros(out_features))
        self.momentum = momentum
        self.quantile = quantile

    def update(self, batch_activation):
        """
        Update the running quantile using the current batch's activations.
        
        Args:
            batch_activation (Tensor): Activations of shape [batch_size, out_features].
        """

        # Compute the quantile (e.g., 98th percentile) along the batch dimension.
        batch_q = torch.quantile(batch_activation.float(), q=self.quantile, dim=0)
        # Update the running quantile using an exponential moving average in a no-grad context.
        with torch.no_grad():
            updated = self.momentum * batch_q + (1 - self.momentum) * self.running_quantile
            self.running_quantile.copy_(updated)


class DGProjectionWithRunningQuantile(nn.Module):
    def __init__(self, in_features: int, out_features: int, momentum: float = 0.1, quantile: float = 0.97):
        """
        Enforces that each neuron's output (after softmax) is activated (set to 1)
        only if its probability exceeds the running quantile (e.g., 98th percentile)
        across previous batches.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.running_quantile = RunningActivationQuantile(out_features, momentum=momentum, quantile=quantile)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)  # Shape: [batch_size, out_features]
        # Replace logits with softmaxed probabilities.
        probs = F.softmax(logits, dim=1)
        
        # Update the running quantile using the current batch's probabilities.
        self.running_quantile.update(probs)
        
        # Get the current running threshold per neuron and broadcast to match batch size.
        threshold = self.running_quantile.running_quantile.unsqueeze(0)  # Shape: [1, out_features]
        
        # Create a binary mask: 1 if probability is above threshold, 0 otherwise.
        mask = (probs > threshold).float()
        
        # Use a straight-through estimator: forward pass uses the hard mask,
        # but gradients flow as if the operation were the identity.
        output = mask + probs - probs.detach()
        return output

import gymnasium as gym
import numpy as np
class HipposlamEncoder(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)


        if cfg.Hippo_n_feature:
            self.Hippo_n_feature = cfg.Hippo_n_feature
        else:
            self.Hippo_n_feature = 64
            log.info("hippo n feature not set, using default: {self.Hippo_n_feature}")
        self.depth_sensor=getattr(cfg, "depth_sensor", False)
        # self.depth_sensor=False
        log.info(self.depth_sensor)
        if self.depth_sensor:
            # obs_depth = obs_space["obs"]
            # obs_depth.shape[0]=1
            log.info(f"using depth sensor {self.depth_sensor}")
            self.depth_encoder = DepthEncoder(cfg, size=10)
        else:
            self.depth_sensor = False

        log.debug(f"original obs space: {obs_space['obs']}")
        obs_cnn=obs_space["obs"]

        obs_cnn=gym.spaces.Box(low=0, high=255, shape=(3,cfg.res_h, cfg.res_w), dtype=np.uint8)
        self.basic_encoder = make_img_encoder(cfg, obs_cnn)
        self.encoder_out_size = self.basic_encoder.get_out_size()
        
        self.with_number_instruction = cfg.with_number_instruction
        self.number_instruction_coef = getattr(cfg, "number_instruction_coef", 1)
        if self.with_number_instruction:
            # repurposed it to encode map number
            self.instructions_lstm_units = 3
        else:
            # same as IMPALA paper
            self.embedding_size = 20
            self.instructions_lstm_units = 64
            self.instructions_lstm_layers = 1

            padding_idx = 0
            self.word_embedding = nn.Embedding(
                num_embeddings=DMLAB_VOCABULARY_SIZE, embedding_dim=self.embedding_size, padding_idx=padding_idx
            )

            self.instructions_lstm = nn.LSTM(
                input_size=self.embedding_size,
                hidden_size=self.instructions_lstm_units,
                num_layers=self.instructions_lstm_layers,
                batch_first=True,
            )

        # learnable initial state?
        # initial_hidden_values = torch.normal(0, 1, size=(self.instructions_lstm_units, ))
        # self.lstm_h0 = nn.Parameter(initial_hidden_values, requires_grad=True)
        # self.lstm_c0 = nn.Parameter(initial_hidden_values, requires_grad=True)

        self.encoder_out_size += self.instructions_lstm_units
        log.info("DMLab policy head output size: %r", self.encoder_out_size)

        if cfg.DG_lr:
            self.dg_lr = getattr(cfg, "DG_lr", 0.001)
        
            self.DG_projection = DGProjection(self.encoder_out_size,cfg.Hippo_n_feature,self.dg_lr)
        elif cfg.DG_temperature:
            self.temperature = getattr(cfg, "DG_temperature", 1)
            self.DG_projection = DGProjection_simple_top1(self.encoder_out_size,cfg.Hippo_n_feature,self.temperature)
        elif cfg.DG_batch_q:
            self.DG_projection = DGProjectionWithRunningQuantile(self.encoder_out_size,cfg.Hippo_n_feature)
        elif cfg.DG_softmax:
            self.DG_projection = DGProjection_simple_softmax(self.encoder_out_size,cfg.Hippo_n_feature)
        else:
            self.DG_projection = nn.Linear(self.encoder_out_size,cfg.Hippo_n_feature)

        if cfg.DG_name == "log_softmax":
            self.DG_projection = DGProjection_log_softmax(self.encoder_out_size,cfg.Hippo_n_feature)
        elif cfg.DG_name == "linear_relu":
            self.DG_projection = DGProjection_relu(self.encoder_out_size,cfg.Hippo_n_feature)
        elif cfg.DG_name == "batch_novelty":
            self.dg_detect = getattr(cfg, "DG_detect", 0.1)
            if not self.dg_detect:
                print("getattr doesn't behave like you think, setting dg_detect to 0.1")
                self.dg_detect = 0.1
            self.dg_novelty = getattr(cfg, "DG_novelty", 0.4)
            if not self.dg_novelty:
                print("getattr doesn't behave like you think, setting dg_novelty to 0.4")
                self.dg_novelty = 0.4
            self.DG_projection = DGProjectionBatchNovelty(self.encoder_out_size,cfg.Hippo_n_feature,self.dg_detect,self.dg_novelty)
        elif cfg.DG_name == "batchnorm_relu":
            intercept=getattr(cfg, "DG_BN_intercept",2)
            self.DG_projection = DGProjection_batchnorm_relu(self.encoder_out_size,cfg.Hippo_n_feature,intercept=intercept)
###
        self.fix_DG=getattr(cfg, "fix_DG", False)
        if self.fix_DG:
            log.info('fix DG weights')
            # Double-check that the encoder parameters are frozen.
            for param in self.DG_projection.parameters():
                param.requires_grad = False

            self.DG_projection.eval()
        else:
            log.info('trainable DG encoder')



###
        self.DG_load_path = getattr(cfg, "DG_load_path", None)
        if self.DG_load_path:
            # Load the checkpoint.
            devicename = cfg.device
            if devicename=='gpu': devicename='cuda'
            checkpoint = torch.load(self.DG_load_path, map_location=devicename)

            full_state_dict = checkpoint["model"]

            # Filter out only the keys for the encoder.
            DG_state_dict = {k.replace("encoder.DG_projection.", ""): v for k, v in full_state_dict.items() if k.startswith("encoder.DG_projection.")}


            # Load the encoder state dict into the new encoder instance.
            self.DG_projection.load_state_dict(DG_state_dict)

            if True: #cfg.fix_encoder_when_load:
                log.info('fix DG weights')
                # Double-check that the encoder parameters are frozen.
                for param in self.DG_projection.parameters():
                    param.requires_grad = False

                self.DG_projection.eval()  # Make sure the encoder is in evaluation mode.
            else:
                log.info('trainable loaded DG encoder')




###
        tmp_out_size = cfg.Hippo_n_feature

        if cfg.core_name.startswith("SeqDense"):#"Gate":
            self.n_dense_feature = getattr(cfg, "N_dense_feature", 16)
            if not self.n_dense_feature:
                print("getattr doesn't behave like you think, setting n_dense_feature to 16")
                self.n_dense_feature = 16
            self.dense = nn.Linear(self.encoder_out_size, self.n_dense_feature)
            tmp_out_size += self.n_dense_feature

        bypass_features = 0
        bypass_features = self.encoder_out_size
        if hasattr(cfg,'depth_sensor'):
            log.info(f"denpth_sensor {cfg.depth_sensor}")
            if self.depth_sensor:
                log.info(f"denpth_sensor {self.depth_sensor}")
                bypass_features = self.depth_encoder.get_out_size() + self.instructions_lstm_units
            
        self.bypass=False
        if cfg.core_name.startswith("Bypass"):#"Gate":
            self.bypass=True
            tmp_out_size += bypass_features
            log.info(f'using bypass, dim {bypass_features}')

        self.encoder_out_size = tmp_out_size
        self.cpu_device = torch.device("cpu")

        # log.info("=================================== memory=========================")
        # log.info(torch.cuda.memory_allocated())

    def model_to_device(self, device): 
        self.to(device)
        if self.with_number_instruction:
            return
        self.word_embedding.to(self.cpu_device)
        self.instructions_lstm.to(self.cpu_device)

    def device_for_input_tensor(self, input_tensor_name: str) -> torch.device:
        if input_tensor_name == DMLAB_INSTRUCTIONS:
            return self.cpu_device
        return model_device(self)

    def type_for_input_tensor(self, input_tensor_name: str) -> torch.dtype:
        if input_tensor_name == DMLAB_INSTRUCTIONS:
            return torch.int64
        return torch.float32

    def forward(self, obs_dict):
        # obs_cnn = obs_dict["obs"].copy()
        if self.depth_sensor:
            obs_cnn = obs_dict["obs"][:,:3,:,:]
        else:
            obs_cnn = obs_dict["obs"][:,:,:,:]
        x = self.basic_encoder(obs_cnn)

        if self.with_number_instruction:
            instr = obs_dict[DMLAB_INSTRUCTIONS]
            last_outputs = torch.nn.functional.one_hot(instr.squeeze(1)-1,num_classes=3)*self.number_instruction_coef
            
            # log.info(last_outputs)

        else:

            with torch.no_grad():
                instr = obs_dict[DMLAB_INSTRUCTIONS]
                instr_lengths = (instr != 0).sum(axis=1)
                instr_lengths = torch.clamp(instr_lengths, min=1)
                max_instr_len = torch.max(instr_lengths).item()
                instr = instr[:, :max_instr_len]

            instr_embed = self.word_embedding(instr)
            instr_packed = torch.nn.utils.rnn.pack_padded_sequence(
                instr_embed,
                instr_lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            rnn_output, _ = self.instructions_lstm(instr_packed)
            rnn_outputs, sequence_lengths = torch.nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)

            first_dim_idx = torch.arange(rnn_outputs.shape[0])
            last_output_idx = sequence_lengths - 1
            last_outputs = rnn_outputs[first_dim_idx, last_output_idx]

        last_outputs = last_outputs.to(x.device)  # for some reason this is very slow


        x = torch.cat((x, last_outputs), dim=1)

        tmp_out = self.DG_projection(x)
        # log.info(tmp_out)
        if self.depth_sensor:
            depth_out = self.depth_encoder(obs_dict['obs'][:,-1:,:,:])
            depth_out = depth_out.view(obs_dict['obs'].size(0),-1)
            bypass_out = torch.cat((depth_out,last_outputs), dim=1)
        else:
            bypass_out = x

        if self.bypass:
            tmp_out = torch.cat((tmp_out,bypass_out), dim=1)
        elif hasattr(self,'dense'):
            dense_out = self.dense(x)
            tmp_out = torch.cat((tmp_out,dense_out), dim=1)

        return tmp_out

    def get_out_size(self) -> int:
        return self.encoder_out_size


def make_hipposlam_encoder(cfg: Config, obs_space: ObsSpace) -> Encoder:
    if cfg.encoder_name=="Default":
        return DmlabEncoder(cfg,obs_space)
    return HipposlamEncoder(cfg, obs_space)


from torch.nn.utils.rnn import PackedSequence

class FixedRNNSequenceCore(ModelCore):
    def __init__(self, cfg, input_size):
        """
        Args:
            cfg: Configuration object with attributes Hippo_R, Hippo_L, Hippo_n_feature.
            input_size (int): Dimensionality of the input observation. (Here we assume it matches Hippo_n_feature.)
        """
        super().__init__(cfg)
        # Use configuration or defaults.
        self.R = getattr(cfg, 'Hippo_R', 8)
        self.L = getattr(cfg, 'Hippo_L', 48)
        self.Hippo_n_feature = getattr(cfg, 'Hippo_n_feature', 64)

        if self.Hippo_n_feature != input_size:
            raise Warning(f"hippo_n_feature{self.Hippo_n_feature } does not match input size {input_size}")
        
        # The total register length.
        self.expanded_length = self.R + self.L - 1  
        # The flattened hidden state dimension.
        self.core_output_size = self.Hippo_n_feature * self.expanded_length
        self.n_feature = self.Hippo_n_feature
        self.hidden_size = self.n_feature * self.expanded_length
        
        # Create a one-layer RNN with ReLU activation.
        # (We use ReLU so that if inputs and state are nonnegative, the activation is effectively identity.)
        self.rnn = nn.RNN(input_size=self.n_feature, 
                          hidden_size=self.hidden_size,
                          num_layers=1, 
                          nonlinearity='relu',
                          batch_first=False)
        
        # Create fixed weight matrices.
        # weight_ih: shape (hidden_size, input_size)
        W_ih = torch.zeros(self.hidden_size, self.n_feature)
        for i in range(self.n_feature):
            for j in range(self.R):
                row = i * self.expanded_length + j
                W_ih[row, i] = 1.0

        # weight_hh: shape (hidden_size, hidden_size)
        W_hh = torch.zeros(self.hidden_size, self.hidden_size)
        for i in range(self.n_feature):
            for j in range(1, self.expanded_length):
                row = i * self.expanded_length + j
                col = i * self.expanded_length + (j - 1)
                W_hh[row, col] = 1.0

        # Assign the fixed weights and set biases to zero.
        with torch.no_grad():
            self.rnn.weight_ih_l0.copy_(W_ih)
            self.rnn.weight_hh_l0.copy_(W_hh)
            self.rnn.bias_ih_l0.zero_()
            self.rnn.bias_hh_l0.zero_()
        
        # Freeze the weights so that they are not updated during training.
        for param in self.rnn.parameters():
            param.requires_grad = False

    def forward(self, head_output, rnn_states):
        """
        Args:
            head_output: Either a Tensor of shape (B, input_size) or a PackedSequence.
            rnn_states: Tensor of shape (B, core_output_size) representing the flattened recurrent state.
        Returns:
            Tuple (core_output, new_rnn_states)
        """
        # Ensure hidden state is contiguous before passing to RNN.
        h0 = rnn_states.unsqueeze(0).contiguous()

        if isinstance(head_output, PackedSequence):
            output, new_hidden = self.rnn(head_output, h0)
            new_hidden = new_hidden.squeeze(0)  # (B, core_output_size)
            return output, new_hidden
        else:
            head_output = head_output.unsqueeze(0)  # (1, B, input_size)
            output, new_hidden = self.rnn(head_output, h0)
            new_hidden = new_hidden.squeeze(0)       # (B, core_output_size)
            output = output.squeeze(0)               # (B, core_output_size)
            return output, new_hidden
        

from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pad_packed_sequence,pack_padded_sequence


class FixedRNNWithBypassCore(ModelCore):
    def __init__(self, cfg, input_size):
        """
        Args:
            cfg: Configuration object with attributes Hippo_R, Hippo_L, Hippo_n_feature.
            input_size (int): Dimensionality of the input observation. 
                              The first Hippo_n_feature dimensions are fed into the fixed RNN,
                              and the remaining (if any) are passed through as bypass features.
        """
        super().__init__(cfg)
        # Use configuration or defaults.
        self.R = getattr(cfg, 'Hippo_R', 8)
        self.L = getattr(cfg, 'Hippo_L', 48)
        self.Hippo_n_feature = getattr(cfg, 'Hippo_n_feature', 64)

        if input_size < self.Hippo_n_feature:
            raise Warning(f"Input size {input_size} must be at least Hippo_n_feature ({self.Hippo_n_feature})")
        self.bypass_size = input_size - self.Hippo_n_feature
        log.debug("bypass size: {self.bypass_size}")
        # The total register length.
        self.expanded_length = self.R + self.L - 1  
        # The flattened hidden state dimension (RNN core output).
        self.core_output_size = self.Hippo_n_feature * self.expanded_length + self.bypass_size
        self.n_feature = self.Hippo_n_feature
        self.hidden_size = self.n_feature * self.expanded_length

        # Create a one-layer RNN with ReLU activation.
        self.rnn = nn.RNN(input_size=self.n_feature, 
                          hidden_size=self.hidden_size,
                          num_layers=1, 
                          nonlinearity='relu',
                          batch_first=False)
        
        # Create fixed weight matrices.
        # weight_ih: shape (hidden_size, n_feature)
        W_ih = torch.zeros(self.hidden_size, self.n_feature)
        for i in range(self.n_feature):
            for j in range(self.R):
                row = i * self.expanded_length + j
                W_ih[row, i] = 1.0

        # weight_hh: shape (hidden_size, hidden_size)
        W_hh = torch.zeros(self.hidden_size, self.hidden_size)
        for i in range(self.n_feature):
            for j in range(1, self.expanded_length):
                row = i * self.expanded_length + j
                col = i * self.expanded_length + (j - 1)
                W_hh[row, col] = 1.0

        # Assign the fixed weights and zero out biases.
        with torch.no_grad():
            self.rnn.weight_ih_l0.copy_(W_ih)
            self.rnn.weight_hh_l0.copy_(W_hh)
            self.rnn.bias_ih_l0.zero_()
            self.rnn.bias_hh_l0.zero_()
        
        # Freeze RNN parameters.
        for param in self.rnn.parameters():
            param.requires_grad = False

    def forward(self, head_output, rnn_states):
        """
        Args:
            head_output: Either a Tensor of shape (B, input_size) or a PackedSequence.
            rnn_states: Tensor of shape (B, core_output_size) representing the flattened recurrent state.
        Returns:
            Tuple (concat_output, new_rnn_states) where:
              - concat_output is the concatenation of the fixed RNN output and the bypass features.
              - new_rnn_states is the updated recurrent state.
        """
        # Prepare initial hidden state for RNN.
        # log.info(rnn_states.size())
        h0 = rnn_states.unsqueeze(0)[:,:, :self.hidden_size].contiguous()
        
        
        if isinstance(head_output, PackedSequence):
            # For PackedSequence, work on the underlying data.
            # Split into RNN and bypass parts.
            rnn_data = head_output.data[:, :self.n_feature]
            bypass_data = head_output.data[:, self.n_feature:] if self.bypass_size > 0 else None

            # Create a PackedSequence for the RNN input.
            rnn_packed = PackedSequence(rnn_data,
                                        head_output.batch_sizes,
                                        head_output.sorted_indices,
                                        head_output.unsorted_indices)
            # Run the RNN.
            rnn_output_packed, new_hidden = self.rnn(rnn_packed, h0)
            new_hidden = new_hidden.squeeze(0)  # shape: (B, core_output_size)
            
            # If bypass features exist, concatenate them.
            if bypass_data is not None:
                # Concatenate along the feature dimension.
                concatenated_data = torch.cat([rnn_output_packed.data, bypass_data], dim=1)

                bypass_data_packed = PackedSequence(bypass_data,
                                        head_output.batch_sizes,
                                        head_output.sorted_indices,
                                        head_output.unsorted_indices)
                # Assume 'packed' is your PackedSequence and you used batch_first=True when packing.
                padded, lengths = pad_packed_sequence(bypass_data_packed, batch_first=True)

                # For each sequence in the batch, pick the last valid time step.
                # lengths is a tensor of the original sequence lengths.
                last_inputs = padded[torch.arange(padded.size(0)), lengths - 1, :]
                concatenated_data_hidden = torch.cat([new_hidden.data, last_inputs], dim=1)

                # # Compute indices in the packed data that correspond to the last time step of each sequence.
                # last_indices = head_output.batch_sizes.cumsum(0) - 1

                # # Use these indices to index into the bypass_data tensor.
                # last_bypass = bypass_data[last_indices,:]

                # # If the sequences were originally unsorted, restore the original order:
                # last_bypass = last_bypass[head_output.unsorted_indices,:]
                # concatenated_data_hidden = torch.cat([new_hidden.data, last_bypass], dim=1)

            else:
                concatenated_data = rnn_output_packed.data
                concatenated_data_hidden = new_hidden.data

            # Build a new PackedSequence with the concatenated data.
            concat_output = PackedSequence(concatenated_data,
                                           rnn_output_packed.batch_sizes,
                                           rnn_output_packed.sorted_indices,
                                           rnn_output_packed.unsorted_indices)
            
            concat_hidden = PackedSequence(concatenated_data_hidden,
                                           rnn_output_packed.batch_sizes,
                                           rnn_output_packed.sorted_indices,
                                           rnn_output_packed.unsorted_indices)
            return concat_output, concat_hidden
        else:
            # For Tensor input.
            # Split the input into RNN and bypass parts.
            rnn_input = head_output[:, :self.n_feature]  # shape: (B, n_feature)
            bypass_output = head_output[:, self.n_feature:] if self.bypass_size > 0 else None
            
            # Add sequence dimension for the RNN.
            rnn_input = rnn_input.unsqueeze(0)  # shape: (1, B, n_feature)
            rnn_output, new_hidden = self.rnn(rnn_input, h0)
            new_hidden = new_hidden.squeeze(0)   # shape: (B, core_output_size)
            rnn_output = rnn_output.squeeze(0)     # shape: (B, core_output_size)
            
            # Concatenate the RNN output with bypass features.
            if bypass_output is not None:
                concat_output = torch.cat([rnn_output, bypass_output], dim=1)
                concat_hidden = torch.cat([new_hidden, bypass_output], dim=1)
            else:
                concat_output = rnn_output

                concat_hidden = new_hidden
            
            return concat_output, concat_hidden






class BypassLSTMCore(ModelCore):
    def __init__(self, cfg, input_size, hidden_size=None, num_layers=1, bidirectional=False):
        """
        Args:
            cfg: Configuration object (for API compatibility).
            input_size (int): Dimensionality of the input observation.
            hidden_size (int): Hidden size for the LSTM. Defaults to Hippo_n_feature * (R+L-1).
            num_layers (int): Number of LSTM layers.
            bidirectional (bool): Whether to use a bidirectional LSTM.
        """
        super().__init__(cfg)
        self.Hippo_n_feature = getattr(cfg, 'Hippo_n_feature', 16)
        self.R = getattr(cfg, 'Hippo_R', 8)
        self.L = getattr(cfg, 'Hippo_L', 64)
        if input_size < self.Hippo_n_feature:
            raise Warning(
                f"Input size {input_size} must be at least Hippo_n_feature ({self.Hippo_n_feature})"
            )
        self.bypass_size = input_size - self.Hippo_n_feature

        # LSTM hyperparameters
        self.hidden_size = getattr(cfg, "bypassLSTM_hidden",167) #or (self.Hippo_n_feature * (self.R + self.L - 1))
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dirs = 2 if bidirectional else 1

        # Define LSTM core
        self.lstm = nn.LSTM(
            input_size=self.Hippo_n_feature,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=False,
            bidirectional=self.bidirectional,
        )

        # Output dimension: LSTM output per time = hidden_size * dirs
        self.core_output_size = self.hidden_size * self.dirs
        self.total_output_size = self.core_output_size + self.bypass_size

        # Flat state size: h and c, both packed, plus bypass
        self.state_hidden_size = self.num_layers * self.dirs * self.hidden_size
        self.state_size = self.state_hidden_size * 2 + self.bypass_size

    def forward(self, head_output, rnn_states):
        """
        head_output: Tensor (B, input_size) or PackedSequence
        rnn_states: Tensor (B, state_size) containing [flat_h; flat_c; bypass]
        Returns (output, new_rnn_states Tensor)
        """
        # Prepare shapes
        if isinstance(head_output, PackedSequence):
            B = head_output.batch_sizes[0] if hasattr(head_output, 'batch_sizes') else None
            device = head_output.data.device
        else:
            B = head_output.size(0)
            device = head_output.device

        # Unpack previous state
        flat_h = rnn_states[:, :self.state_hidden_size]
        flat_c = rnn_states[:, self.state_hidden_size:2*self.state_hidden_size]
        if self.bypass_size > 0:
            prev_bypass = rnn_states[:, -self.bypass_size:]
        else:
            prev_bypass = None
        # reshape to (layers*dirs, B, hidden)
        h0 = flat_h.view(B, self.num_layers * self.dirs, self.hidden_size).permute(1,0,2).contiguous()
        c0 = flat_c.view(B, self.num_layers * self.dirs, self.hidden_size).permute(1,0,2).contiguous()

        if isinstance(head_output, PackedSequence):
            # Split into core and bypass
            core_data = head_output.data[:, :self.Hippo_n_feature]
            bypass_data = head_output.data[:, self.Hippo_n_feature:] if self.bypass_size>0 else None
            core_packed = PackedSequence(core_data,
                                         head_output.batch_sizes,
                                         head_output.sorted_indices,
                                         head_output.unsorted_indices)
            # LSTM forward
            out_packed, (h_n, c_n) = self.lstm(core_packed, (h0, c0))
            padded_out, lengths = pad_packed_sequence(out_packed)
            # Bypass sequence padded
            padded_full, _ = pad_packed_sequence(head_output)
            if self.bypass_size>0:
                bypass_seq = padded_full[:, :, self.Hippo_n_feature:]
                total_seq = torch.cat([padded_out, bypass_seq], dim=2)
            else:
                total_seq = padded_out
            # repack output
            output = pack_padded_sequence(total_seq, lengths, enforce_sorted=False)

            # Build new flat state
            flat_hn = h_n.permute(1,0,2).contiguous().view(B, -1)
            flat_cn = c_n.permute(1,0,2).contiguous().view(B, -1)
            # last valid bypass
            if self.bypass_size>0:
                # padded_full: T, B, input_size
                idx = lengths - 1
                last_bypass = padded_full[idx, torch.arange(B), self.Hippo_n_feature:]
                new_state = torch.cat([flat_hn, flat_cn, last_bypass], dim=1)
            else:
                new_state = torch.cat([flat_hn, flat_cn], dim=1)

            return output, new_state

        else:
            # single-step tensor
            core_in = head_output[:, :self.Hippo_n_feature].unsqueeze(0)  # 1, B, F
            out, (h_n, c_n) = self.lstm(core_in, (h0, c0))
            core_out = out.squeeze(0)
            if self.bypass_size>0:
                bypass = head_output[:, self.Hippo_n_feature:]
                total_out = torch.cat([core_out, bypass], dim=1)
                last_bypass = bypass
            else:
                total_out = core_out
                last_bypass = None

            # new flat state
            flat_hn = h_n.permute(1,0,2).contiguous().view(B, -1)
            flat_cn = c_n.permute(1,0,2).contiguous().view(B, -1)
            if self.bypass_size>0:
                new_state = torch.cat([flat_hn, flat_cn, last_bypass], dim=1)
            else:
                new_state = torch.cat([flat_hn, flat_cn], dim=1)

            return total_out, new_state

    def get_out_size(self) -> int:
        return self.core_output_size + self.bypass_size

    def get_state_size(self) -> int:
        return self.state_size





class SimpleSequenceCore(ModelCore):
    def __init__(self, cfg, input_size):
        """
        Args:
            input_dim (int): Dimensionality of the input observation.
            R (int): Number of positions to update with the feature value.
            L (int): Number of time steps over which the shift register is built.
        """
        super().__init__(cfg)
        self.dim = input_size
        self.cfg = cfg
        if cfg.Hippo_R:
            self.R = cfg.Hippo_R
        else:
            self.R = 8
            log.info("R not set, using default: {self.R}")
        if cfg.Hippo_L:
            self.L = cfg.Hippo_L
        else:
            self.L = 48
            log.info("L not set, using default: {self.R}")
        if cfg.Hippo_n_feature:
            self.Hippo_n_feature = cfg.Hippo_n_feature
        else:
            self.Hippo_n_feature = 64
            log.info("hippo n feature not set, using default: {self.Hippo_n_feature}")
        
        # self.linear = nn.Linear(input_size, self.Hippo_n_feature)  # Map input to 64-dimensional features.
        self.expanded_length = self.R + self.L - 1  # Total length of the shift register.

        # self.rnn_states = torch.zeros(batch_size,self.Hippo_n_feature*self.expanded_length, device=device)
        self.core_output_size = self.Hippo_n_feature * self.expanded_length

    def forward(self, head_output, rnn_states):
        """
        Args:
            head_output: Either a Tensor of shape (B, input_dim) from the encoder 
                        or a PackedSequence.
            rnn_states: Tensor of shape (B, Hippo_n_feature * expanded_length)
                        representing the flattened recurrent state.
                        
        Returns:
            Tuple (core_output, new_rnn_states), both of shape 
            (B, Hippo_n_feature * expanded_length) if the input is a single time step.
            (If a sequence is provided, the time dimension is preserved.)
        """
        # Determine if head_output is a PackedSequence or a plain tensor.
        # this supposedly also means inference mode or training mode
        is_seq = not torch.is_tensor(head_output)
        if is_seq:
            # log.info('==========packed sequence')
            # Unpack the sequence; padded_out shape: (T, B, input_dim)
            _, batch_sizes, sorted_indices, unsorted_indices = head_output
            padded_out, lengths = nn.utils.rnn.pad_packed_sequence(head_output)

            # batch_indices=batch_sizes-1

            # unsorted_indices = getattr(head_output, 'unsorted_indices', None)

            features = padded_out
            # log.info(features.size())
            # log.info(batch_sizes[:5])
        else:
            # log.info('==========plain tensor')
            # If a plain tensor is provided, add a time dimension.
            # head_output = head_output.unsqueeze(0)  # now (1, B, input_dim)
            # lengths = [1]* head_output.size(0)
            features = head_output    
            features=features.unsqueeze(0)

            batch_sizes=features.size(1)

        rnn_states = rnn_states.unsqueeze(0)
        rnn_states = rnn_states.view(rnn_states.size(0), rnn_states.size(1),
                                    self.Hippo_n_feature, self.expanded_length).contiguous()
        output = torch.empty((features.size(0),features.size(1),self.Hippo_n_feature, self.expanded_length),device=features.device)


        new_rnn_states = rnn_states.clone()

        ### tried to fast propagate hipposeq but it wouldn't work 
        # for i in range(len(lengths)):
        #     # rnn_states = rnn_states[:,i,:,:].roll(shifts=lengths[i], dims=-1).contiguous()
        #     # # propagate hipposeq
        #     # rnn_states[:, i, :, :lengths[i]] = 0

        #     tmp_states= F.pad(rnn_states[:,i,:,:],(lengths[i],0)) #dim (0,n_feature,n_expanded)

        #     kernel=F.pad(torch.ones(self.R,device=rnn_states.device),(0,self.R-1))

        #     injection=F.conv1d(features[:,i,:].permute(1,0).unsqueeze(0),kernel.unsqueeze(0).unsqueeze(0))


        #     rnn_states[:, i, :, :] = rnn_states[:, i, :, :] + injection

        #     # output (time, B, Hippo_n_feature, expanded_length)
        #     output[i,:,:,:]=new_rnn_states[0, :, :, :]
        #     rnn_states=new_rnn_states
        if is_seq:
            for i in range(features.size(0)):

                # Here, state_dim should equal self.Hippo_n_feature * expanded_length.
                # Reshape to 4D so we can perform our custom shift and injection.
                # New shape: (time, B, Hippo_n_feature, expanded_length)

                tmpind = sorted_indices[:batch_sizes[i]]

                # Shift the register one step along the last dimension.
                tmp_rnn_states = new_rnn_states[:,tmpind,:,:].roll(shifts=1, dims=-1).contiguous()
                # Zero out the newly empty slot.
                tmp_rnn_states[:, :, :, 0] = 0


                # Inject the current features into the first R positions.
                # features: (B, Hippo_n_feature) --> unsqueeze to (1, B, Hippo_n_feature, 1)
                # then expand to (1, B, Hippo_n_feature, R)
                injection = features[i,tmpind,:].unsqueeze(0).unsqueeze(-1).expand(1,
                                                                        -1, -1, self.R).contiguous()
                    
                # log.info(tmp_rnn_states.size())
                # log.info(injection.size())
                # log.info('======size_above======')


                tmp_rnn_states[:, :, :, :self.R] = tmp_rnn_states[:, :, :, :self.R] + injection


                new_rnn_states[:, tmpind, :, : ] = tmp_rnn_states[:, :, :, :]
                # output (time, B, Hippo_n_feature, expanded_length)
                output[i,:,:,:]=new_rnn_states[0, :, :, :]
        else:



            # Shift the register one step along the last dimension.
            tmp_rnn_states = new_rnn_states[:,:,:,:].roll(shifts=1, dims=-1).contiguous()
            # Zero out the newly empty slot.
            tmp_rnn_states[:, :, :, 0] = 0


            # Inject the current features into the first R positions.
            # features: (B, Hippo_n_feature) --> unsqueeze to (1, B, Hippo_n_feature, 1)
            # then expand to (1, B, Hippo_n_feature, R)
            injection = features[0,:].unsqueeze(-1).expand(tmp_rnn_states.size(0),
                                                                    -1, -1, self.R).contiguous()
            

            tmp_rnn_states[:, :, :, :self.R] = tmp_rnn_states[:, :, :, :self.R] + injection



            new_rnn_states[:, :, :, : ] = tmp_rnn_states[:, :, :, :]
            # output (time, B, Hippo_n_feature, expanded_length)
            output[0,:,:,:]=new_rnn_states[0, :, :, :]



        # Flatten new_rnn_states back to 3D: (time, B, Hippo_n_feature * expanded_length)
        new_rnn_states = new_rnn_states.view(new_rnn_states.size(0),
                                            new_rnn_states.size(1),
                                            self.Hippo_n_feature * self.expanded_length).contiguous()
        
        output = output.view(features.size(0),features.size(1),self.Hippo_n_feature * self.expanded_length)

        # If we added a time dimension for a single step, remove it for output consistency.
        if not is_seq:
            x = output.squeeze(0)  # shape: (B, Hippo_n_feature * expanded_length)
            new_rnn_states = new_rnn_states.squeeze(0)
        else:
            # log.info(new_rnn_states.size())
            new_rnn_states = new_rnn_states.squeeze(0)
            x =nn.utils.rnn.pack_padded_sequence(output, lengths,  enforce_sorted=False)
            # x = output  # Preserve time dimension if multiple steps provided

        # log.info(x.size())

        return x, new_rnn_states

class SimpleSequenceWithBypassCore(ModelCore):
    def __init__(self, cfg, input_size):
        """
        Args:
            cfg: Configuration object with attributes Hippo_R, Hippo_L, Hippo_n_feature.
            input_size (int): Dimensionality of the input observation.
                              The first Hippo_n_feature dimensions are processed through the core,
                              and the remaining (if any) are treated as bypass features.
        """
        super().__init__(cfg)
        self.R = getattr(cfg, 'Hippo_R', 8)
        self.L = getattr(cfg, 'Hippo_L', 48)
        self.Hippo_n_feature = getattr(cfg, 'Hippo_n_feature', 64)
        if input_size < self.Hippo_n_feature:
            raise Warning(
                f"Input size {input_size} must be at least Hippo_n_feature ({self.Hippo_n_feature})"
            )
        self.bypass_size = input_size - self.Hippo_n_feature

        # Total length of the shift register.
        self.expanded_length = self.R + self.L - 1
        # Core (shift register) output dimension.
        self.core_output_size = self.Hippo_n_feature * self.expanded_length
        # Total output dimension when bypass features are concatenated.
        self.total_output_size = self.core_output_size + self.bypass_size

    def forward(self, head_output, rnn_states):
        """
        Args:
            head_output: Either a Tensor of shape (B, input_size) (single time step)
                         or a PackedSequence (multiple time steps).
            rnn_states: Tensor of shape (B, total_output_size) representing the flattened recurrent state.
                        (Only the first core_output_size entries are updated with the shift-register mechanism;
                         bypass features are updated using the most recent input.)
        Returns:
            Tuple (core_output, new_rnn_states) where:
              - core_output has shape (B, total_output_size) if the input is a single time step,
                or is a PackedSequence with the time dimension preserved.
              - new_rnn_states is updated similarly.
        """
        # Case: head_output is a PackedSequence (multiple time steps)
        if isinstance(head_output, PackedSequence):
            # Unpack the sequence.
            # head_output is a namedtuple with (data, batch_sizes, sorted_indices, unsorted_indices)
            _, batch_sizes, sorted_indices, unsorted_indices = head_output
            padded, lengths = nn.utils.rnn.pad_packed_sequence(head_output)
            T, B, input_size = padded.shape  # T: time steps, B: max batch size

            # Separate core state and bypass part from the recurrent state.
            core_state = rnn_states[:, :self.core_output_size].view(B, self.Hippo_n_feature, self.expanded_length)
            # We add a time dimension (of size 1) for in-loop updates.
            new_core_state = core_state.unsqueeze(0).clone()

            # We'll store the flattened core outputs for each time step.
            out_core = torch.empty((T, B, self.core_output_size), device=padded.device)

            # Process each time step updating only the valid (sorted) indices.
            for t in range(T):
                valid_idx = sorted_indices[:batch_sizes[t]]
                # Extract the current core input for valid batch indices.
                curr_core = padded[t, valid_idx, :self.Hippo_n_feature]
                # Update the core state for these indices:
                # Roll the shift register by one.
                tmp_state = new_core_state[:, valid_idx, :, :].roll(shifts=1, dims=-1)
                # Zero the newly available slot.
                tmp_state[:, :, :, 0] = 0
                # Inject the current core features into the first R positions.
                injection = curr_core.unsqueeze(0).unsqueeze(-1).expand(1, -1, -1, self.R)
                tmp_state[:, :, :, :self.R] += injection
                # Update the new core state for valid indices.
                new_core_state[:, valid_idx, :, :] = tmp_state
                # Save the flattened core state (for all batches) at time t.
                out_core[t] = new_core_state[0].view(B, self.core_output_size)

            # Compute the final core part of the new rnn state.
            final_core = new_core_state[0].view(B, self.core_output_size)
            # For bypass, we do not update it recurrently.
            # Instead, for each sequence we take the last valid bypass input.
            if self.bypass_size > 0:
                last_bypass = []
                for i in range(B):
                    last_bypass.append(padded[lengths[i] - 1, i, self.Hippo_n_feature:].unsqueeze(0))
                last_bypass = torch.cat(last_bypass, dim=0)  # shape: (B, bypass_size)
            else:
                last_bypass = None

            # Form the new recurrent state by concatenating final core and bypass.
            if self.bypass_size > 0:
                new_rnn_state = torch.cat([final_core, last_bypass], dim=1)
            else:
                new_rnn_state = final_core

            # For the output at each time step, the bypass features come directly from the current input.
            # We can concatenate the core output computed in the loop with the bypass part from the padded sequence.
            if self.bypass_size > 0:
                # padded_bypass has shape (T, B, bypass_size)
                padded_bypass = padded[:, :, self.Hippo_n_feature:]
                out_total = torch.cat([out_core, padded_bypass], dim=2)
            else:
                out_total = out_core

            # Repack the output using the original lengths.
            new_output = nn.utils.rnn.pack_padded_sequence(out_total, lengths, enforce_sorted=False)
            return new_output, new_rnn_state

        else:
            # Case: head_output is a plain tensor (single time step)
            B = head_output.size(0)
            core_input = head_output[:, :self.Hippo_n_feature]
            bypass_input = head_output[:, self.Hippo_n_feature:] if self.bypass_size > 0 else None

            # Reshape the core part of the rnn state.
            core_state = rnn_states[:, :self.core_output_size].view(B, self.Hippo_n_feature, self.expanded_length)
            # Roll the shift register and zero the new slot.
            new_core_state = core_state.roll(shifts=1, dims=-1)
            new_core_state[:, :, 0] = 0
            # Inject the current core input into the first R positions.
            injection = core_input.unsqueeze(-1).expand(-1, -1, self.R)
            new_core_state[:, :, :self.R] += injection
            flat_core = new_core_state.view(B, self.core_output_size)
            # The new rnn state combines the updated core state with the current bypass input.
            if self.bypass_size > 0:
                out = torch.cat([flat_core, bypass_input], dim=1)
                new_rnn_state = out
            else:
                out = flat_core
                new_rnn_state = flat_core
            return out, new_rnn_state

    def get_out_size(self) -> int:
        log.debug(f"get out size called: {self.total_output_size}")
        return self.total_output_size
    
def straight_through_binary(x:Tensor,identity=True):
    x_binary=(x>0).float()
    if identity:
        return(x_binary - x.detach() + x)

class SimpleSequenceWithBypassCore_binary(ModelCore):
    def __init__(self, cfg, input_size):
        """
        Args:
            cfg: Configuration object with attributes Hippo_R, Hippo_L, Hippo_n_feature.
            input_size (int): Dimensionality of the input observation.
                              The first Hippo_n_feature dimensions are processed through the core,
                              and the remaining (if any) are treated as bypass features.
        """
        super().__init__(cfg)
        self.R = getattr(cfg, 'Hippo_R', 8)
        self.L = getattr(cfg, 'Hippo_L', 48)
        self.Hippo_n_feature = getattr(cfg, 'Hippo_n_feature', 64)
        if input_size < self.Hippo_n_feature:
            raise Warning(
                f"Input size {input_size} must be at least Hippo_n_feature ({self.Hippo_n_feature})"
            )
        self.bypass_size = input_size - self.Hippo_n_feature

        # Total length of the shift register.
        self.expanded_length = self.R + self.L - 1
        # Core (shift register) output dimension.
        self.core_output_size = self.Hippo_n_feature * self.expanded_length
        # Total output dimension when bypass features are concatenated.
        self.total_output_size = self.core_output_size + self.bypass_size


        self.refractory=getattr(cfg, "refractory", -1)

    def forward(self, head_output, rnn_states):
        """
        Args:
            head_output: Either a Tensor of shape (B, input_size) (single time step)
                         or a PackedSequence (multiple time steps).
            rnn_states: Tensor of shape (B, total_output_size) representing the flattened recurrent state.
                        (Only the first core_output_size entries are updated with the shift-register mechanism;
                         bypass features are updated using the most recent input.)
        Returns:
            Tuple (core_output, new_rnn_states) where:
              - core_output has shape (B, total_output_size) if the input is a single time step,
                or is a PackedSequence with the time dimension preserved.
              - new_rnn_states is updated similarly.
        """
        # Case: head_output is a PackedSequence (multiple time steps)
        if isinstance(head_output, PackedSequence):
            # Unpack the sequence.
            # head_output is a namedtuple with (data, batch_sizes, sorted_indices, unsorted_indices)
            _, batch_sizes, sorted_indices, unsorted_indices = head_output
            padded, lengths = nn.utils.rnn.pad_packed_sequence(head_output)
            T, B, input_size = padded.shape  # T: time steps, B: max batch size

            # Separate core state and bypass part from the recurrent state.
            core_state = rnn_states[:, :self.core_output_size].view(B, self.Hippo_n_feature, self.expanded_length)
            # We add a time dimension (of size 1) for in-loop updates.
            new_core_state = core_state.unsqueeze(0).clone()

            # We'll store the flattened core outputs for each time step.
            out_core = torch.empty((T, B, self.core_output_size), device=padded.device)

            # Process each time step updating only the valid (sorted) indices.
            for t in range(T):
                valid_idx = sorted_indices[:batch_sizes[t]]
                # Extract the current core input for valid batch indices.
                curr_core = padded[t, valid_idx, :self.Hippo_n_feature]
                # Update the core state for these indices:
                # Roll the shift register by one.
                tmp_state = new_core_state[:, valid_idx, :, :].roll(shifts=1, dims=-1)
                # Zero the newly available slot.
                tmp_state[:, :, :, 0] = 0


                if self.refractory!=0:

                    curr_core = straight_through_binary( straight_through_binary(curr_core) - tmp_state[0,:,:,:self.refractory].sum(-1)/self.R)#.values)

                # Inject the current core features into the first R positions.
                injection = curr_core.unsqueeze(0).unsqueeze(-1).expand(1, -1, -1, self.R)

                tmp_state[:, :, :, :self.R] += injection
                # Update the new core state for valid indices.
                new_core_state[:, valid_idx, :, :] = tmp_state
                # Save the flattened core state (for all batches) at time t.
                out_core[t] = new_core_state[0].view(B, self.core_output_size)

            # Compute the final core part of the new rnn state.
            final_core = new_core_state[0].view(B, self.core_output_size)
            # For bypass, we do not update it recurrently.
            # Instead, for each sequence we take the last valid bypass input.
            if self.bypass_size > 0:
                last_bypass = []
                for i in range(B):
                    last_bypass.append(padded[lengths[i] - 1, i, self.Hippo_n_feature:].unsqueeze(0))
                last_bypass = torch.cat(last_bypass, dim=0)  # shape: (B, bypass_size)
            else:
                last_bypass = None

            # Form the new recurrent state by concatenating final core and bypass.
            if self.bypass_size > 0:
                new_rnn_state = torch.cat([final_core, last_bypass], dim=1)
            else:
                new_rnn_state = final_core

            new_rnn_state=straight_through_binary(new_rnn_state)

            # For the output at each time step, the bypass features come directly from the current input.
            # We can concatenate the core output computed in the loop with the bypass part from the padded sequence.
            if self.bypass_size > 0:
                # padded_bypass has shape (T, B, bypass_size)
                padded_bypass = padded[:, :, self.Hippo_n_feature:]
                out_total = torch.cat([out_core, padded_bypass], dim=2)
            else:
                out_total = out_core

            out_total = straight_through_binary(out_total)
            # Repack the output using the original lengths.
            new_output = nn.utils.rnn.pack_padded_sequence(out_total, lengths, enforce_sorted=False)
            return new_output, new_rnn_state

        else:
            # Case: head_output is a plain tensor (single time step)
            B = head_output.size(0)
            core_input = head_output[:, :self.Hippo_n_feature]
            bypass_input = head_output[:, self.Hippo_n_feature:] if self.bypass_size > 0 else None

            # Reshape the core part of the rnn state.
            core_state = rnn_states[:, :self.core_output_size].view(B, self.Hippo_n_feature, self.expanded_length)
            # Roll the shift register and zero the new slot.
            new_core_state = core_state.roll(shifts=1, dims=-1)
            new_core_state[:, :, 0] = 0

            if self.refractory!=0:
                    core_input = straight_through_binary( straight_through_binary(core_input) - new_core_state[:,:,:self.refractory].sum(-1)/self.R)
            # Inject the current core input into the first R positions.
            injection = core_input.unsqueeze(-1).expand(-1, -1, self.R)
            new_core_state[:, :, :self.R] += injection
            flat_core = new_core_state.view(B, self.core_output_size)
            # The new rnn state combines the updated core state with the current bypass input.
            if self.bypass_size > 0:
                out = torch.cat([flat_core, bypass_input], dim=1)
                new_rnn_state = out
            else:
                out = flat_core
                new_rnn_state = flat_core


            
            return straight_through_binary(out), straight_through_binary(new_rnn_state)

    def get_out_size(self) -> int:
        log.debug(f"get out size called: {self.total_output_size}")
        return self.total_output_size






class SimpleSequenceWithBypassCore_outdated(ModelCore):
    def __init__(self, cfg, input_size):
        """
        Args:
            cfg: Configuration object with attributes Hippo_R, Hippo_L, Hippo_n_feature.
            input_size (int): Dimensionality of the input observation.
                              The first Hippo_n_feature dimensions are processed through the core,
                              and the remaining (if any) are treated as bypass features.
        """
        super().__init__(cfg)
        self.R = getattr(cfg, 'Hippo_R', 8)
        self.L = getattr(cfg, 'Hippo_L', 48)
        self.Hippo_n_feature = getattr(cfg, 'Hippo_n_feature', 64)
        if input_size < self.Hippo_n_feature:
            raise Warning(
                f"Input size {input_size} must be at least Hippo_n_feature ({self.Hippo_n_feature})"
            )
        self.bypass_size = input_size - self.Hippo_n_feature

        # Total length of the shift register.
        self.expanded_length = self.R + self.L - 1
        # Core (shift register) output dimension.
        self.core_output_size = self.Hippo_n_feature * self.expanded_length
        # Total output dimension when bypass features are concatenated.
        self.total_output_size = self.core_output_size + self.bypass_size

    def forward(self, head_output, rnn_states):
        """
        Args:
            head_output: Either a Tensor of shape (B, input_size) (single time step)
                         or a PackedSequence (multiple time steps).
            rnn_states: Tensor of shape (B, total_output_size) representing the flattened recurrent state.
                        (Only the first core_output_size entries are updated with the shift-register mechanism;
                         bypass features are updated using the most recent input.)
        Returns:
            Tuple (core_output, new_rnn_states) where:
              - core_output has shape (B, total_output_size) if the input is a single time step,
                or is a PackedSequence with the time dimension preserved.
              - new_rnn_states is updated similarly.
        """
        # Process the case where head_output is a PackedSequence.
        if isinstance(head_output, PackedSequence):
            # Unpack the sequence.
            # Note: head_output is a namedtuple with (data, batch_sizes, sorted_indices, unsorted_indices).
            # We use the sorted_indices and batch_sizes to update only the valid batch entries.
            _, batch_sizes, sorted_indices, unsorted_indices = head_output
            padded, lengths = nn.utils.rnn.pad_packed_sequence(head_output)
            T, B, input_size = padded.shape  # T: time steps, B: max batch size

            # Separate core state and bypass state from the recurrent state.
            core_state = rnn_states[:, :self.core_output_size].view(B, self.Hippo_n_feature, self.expanded_length)
            bypass_state = rnn_states[:, self.core_output_size:] if self.bypass_size > 0 else None

            # We'll add a time dimension for the core state to update it at every time step.
            # new_core_state has shape (1, B, Hippo_n_feature, expanded_length)
            new_core_state = core_state.unsqueeze(0).clone()
            # For bypass, we clone so that we can update valid batch entries.
            if self.bypass_size > 0:
                new_bypass_state = bypass_state.clone()
            else:
                new_bypass_state = None

            # Prepare an output tensor to collect updated (core + bypass) outputs.
            # We'll first build the core part then concatenate the bypass features.
            out_core = torch.empty((T, B, self.core_output_size), device=padded.device)
            out_total = torch.empty((T, B, self.total_output_size), device=padded.device)

            # Process each time step.
            for t in range(T):
                # For the current time step, only the first batch_sizes[t] entries are valid.
                valid_idx = sorted_indices[:batch_sizes[t]]
                # Extract current core input for the valid batch indices.
                curr_core = padded[t, valid_idx, :self.Hippo_n_feature]  # shape: (valid_count, Hippo_n_feature)
                # For bypass, if available, extract the corresponding features.
                if self.bypass_size > 0:
                    curr_bypass = padded[t, valid_idx, self.Hippo_n_feature:]  # shape: (valid_count, bypass_size)

                # Update the core state for valid indices:
                # a) roll the shift register one step to the right (along the last dimension)
                tmp_core = new_core_state[:, valid_idx, :, :].roll(shifts=1, dims=-1)
                # b) zero out the new slot (first position).
                tmp_core[:, :, :, 0] = 0
                # c) inject the current core input into the first R positions.
                # Expand current core input so it can be added to the first R positions.
                injection = curr_core.unsqueeze(0).unsqueeze(-1).expand(1, -1, -1, self.R)
                tmp_core[:, :, :, :self.R] += injection
                # d) update the state.
                new_core_state[:, valid_idx, :, :] = tmp_core

                # Flatten the core state for all batches.
                flat_core = new_core_state[0].view(B, self.core_output_size)

                # For bypass: update only for the valid indices with the current bypass features.
                if self.bypass_size > 0:
                    new_bypass_state[valid_idx] = curr_bypass

                # Combine the core and bypass parts to form the total output.
                if self.bypass_size > 0:
                    combined = torch.cat([flat_core, new_bypass_state], dim=1)  # shape: (B, total_output_size)
                else:
                    combined = flat_core

                # Save the combined state as output at time step t.
                out_total[t] = combined
                # Also keep track of the core portion separately if needed.
                out_core[t] = flat_core

            # After processing the sequence, form the new rnn state.
            new_core_flat = new_core_state[0].view(B, self.core_output_size)
            if self.bypass_size > 0:
                new_rnn_state = torch.cat([new_core_flat, new_bypass_state], dim=1)
            else:
                new_rnn_state = new_core_flat

            # Repack the output sequence. Note that we use the original lengths.
            new_output = nn.utils.rnn.pack_padded_sequence(out_total, lengths, enforce_sorted=False)
            return new_output, new_rnn_state

        else:
            # Processing a single time step (plain tensor of shape (B, input_size)).
            B = head_output.size(0)
            core_input = head_output[:, :self.Hippo_n_feature]
            bypass_input = head_output[:, self.Hippo_n_feature:] if self.bypass_size > 0 else None

            # Reshape the core part of the recurrent state.
            core_state = rnn_states[:, :self.core_output_size].view(B, self.Hippo_n_feature, self.expanded_length)
            # Shift the register: roll along the last dimension and zero the new slot.
            new_core_state = core_state.roll(shifts=1, dims=-1)
            new_core_state[:, :, 0] = 0
            # Inject the current core input into the first R positions.
            injection = core_input.unsqueeze(-1).expand(-1, -1, self.R)
            new_core_state[:, :, :self.R] += injection
            flat_core = new_core_state.view(B, self.core_output_size)

            # For bypass, simply take the current bypass features.
            if self.bypass_size > 0:
                out = torch.cat([flat_core, bypass_input], dim=1)
                new_rnn_state = torch.cat([flat_core, bypass_input], dim=1)
            else:
                out = flat_core
                new_rnn_state = flat_core
            return out, new_rnn_state

    def get_out_size(self) -> int:
        log.debug(f"get out size called: {self.total_output_size}")
        return self.total_output_size


class SimpleSequenceWithBypassCore_no_batch_ordering(ModelCore):
    def __init__(self, cfg, input_size):
        """
        Args:
            cfg: Configuration object with attributes Hippo_R, Hippo_L, Hippo_n_feature.
            input_size (int): Dimensionality of the input observation.
                              The first Hippo_n_feature dimensions are processed through the core,
                              and the remaining (if any) are treated as bypass features.
        """
        super().__init__(cfg)
        self.R = getattr(cfg, 'Hippo_R', 8)
        self.L = getattr(cfg, 'Hippo_L', 48)
        self.Hippo_n_feature = getattr(cfg, 'Hippo_n_feature', 64)
        
        if input_size < self.Hippo_n_feature:
            raise Warning(
                f"Input size {input_size} must be at least Hippo_n_feature ({self.Hippo_n_feature})"
            )
        self.bypass_size = input_size - self.Hippo_n_feature
        
        # Total length of the shift register.
        self.expanded_length = self.R + self.L - 1
        # Core (shift register) output dimension.
        self.core_output_size = self.Hippo_n_feature * self.expanded_length
        # Total output dimension when bypass features are concatenated.
        self.total_output_size = self.core_output_size + self.bypass_size

    def forward(self, head_output, rnn_states):
        """
        Args:
            head_output: Either a Tensor of shape (B, input_size) (single time step)
                         or a PackedSequence (multiple time steps).
            rnn_states: Tensor of shape (B, total_output_size) representing the flattened recurrent state.
                        (Only the first core_output_size entries are updated with the shift-register mechanism;
                        bypass features are updated using the most recent input.)
        Returns:
            Tuple (core_output, new_rnn_states) where:
              - core_output has shape (B, total_output_size) if the input is a single time step,
                or is a PackedSequence with time dimension preserved.
              - new_rnn_states is updated similarly.
        """
        # For core processing, extract and reshape the recurrent state.
        # rnn_states shape: (B, total_output_size) = (B, core_output_size + bypass_size)
        if isinstance(head_output, PackedSequence):
            # Unpack the sequence: padded shape is (T, B, input_size).
            padded, lengths = nn.utils.rnn.pad_packed_sequence(head_output)
            T, B, _ = padded.shape

            # Separate the recurrent state for core processing.
            core_state = rnn_states[:, :self.core_output_size].view(B, self.Hippo_n_feature, self.expanded_length)
            # We will update this core_state for each time step.
            new_core_state = core_state.clone()
            outputs = []
            # Process each time step.
            for t in range(T):
                # Split the input into core and bypass parts.
                curr_core = padded[t, :, :self.Hippo_n_feature]  # shape: (B, Hippo_n_feature)
                # Shift the core state (roll along the last dimension) and zero the new slot.
                new_core_state = new_core_state.roll(shifts=1, dims=-1)
                new_core_state[:, :, 0] = 0
                # Inject current features into the first R positions.
                injection = curr_core.unsqueeze(-1).expand(-1, -1, self.R)
                new_core_state[:, :, :self.R] += injection
                # Flatten the core state.
                flat_core = new_core_state.view(B, self.core_output_size)
                # For bypass, if available, take the current bypass features.
                if self.bypass_size > 0:
                    curr_bypass = padded[t, :, self.Hippo_n_feature:]  # shape: (B, bypass_size)
                    out_t = torch.cat([flat_core, curr_bypass], dim=1)
                else:
                    out_t = flat_core
                outputs.append(out_t.unsqueeze(0))
            
            # Stack outputs along the time dimension.
            outputs = torch.cat(outputs, dim=0)  # shape: (T, B, total_output_size)
            # Repack the sequence.
            new_output = nn.utils.rnn.pack_padded_sequence(outputs, lengths, enforce_sorted=False)
            
            # For updating rnn_states, use the final core state and the last bypass features.
            if self.bypass_size > 0:
                # For each sequence, pick the last valid bypass input.
                last_bypass = []
                for i in range(B):
                    # lengths[i] is the number of time steps for sequence i.
                    last_bypass.append(padded[lengths[i] - 1, i, self.Hippo_n_feature:].unsqueeze(0))
                last_bypass = torch.cat(last_bypass, dim=0)  # (B, bypass_size)
                new_rnn_state = torch.cat([new_core_state.view(B, self.core_output_size), last_bypass], dim=1)
            else:
                new_rnn_state = new_core_state.view(B, self.core_output_size)
            return new_output, new_rnn_state

        else:
            # head_output is a plain tensor of shape (B, input_size) for a single time step.
            B = head_output.size(0)
            core_input = head_output[:, :self.Hippo_n_feature]
            bypass_input = head_output[:, self.Hippo_n_feature:] if self.bypass_size > 0 else None

            # Reshape the core part of the recurrent state.
            core_state = rnn_states[:, :self.core_output_size].view(B, self.Hippo_n_feature, self.expanded_length)
            # Shift the register and zero out the new slot.
            new_core_state = core_state.roll(shifts=1, dims=-1)
            new_core_state[:, :, 0] = 0
            # Inject the current core input into the first R positions.
            injection = core_input.unsqueeze(-1).expand(-1, -1, self.R)
            new_core_state[:, :, :self.R] += injection
            flat_core = new_core_state.view(B, self.core_output_size)
            
            if self.bypass_size > 0:
                out = torch.cat([flat_core, bypass_input], dim=1)
                new_rnn_state = torch.cat([flat_core, bypass_input], dim=1)
            else:
                out = flat_core
                new_rnn_state = flat_core
            return out, new_rnn_state
    def get_out_size(self) -> int:
        log.debug("get out size called: {self.total_output_size}")
        return self.total_output_size

class SS_Bypass_Forget_Core(ModelCore):
    def __init__(self, cfg, input_size):
        """
        Args:
            cfg: Configuration object with attributes Hippo_R, Hippo_L, Hippo_n_feature.
            input_size (int): Dimensionality of the input observation.
                              The first Hippo_n_feature dimensions are processed through the core,
                              and the remaining (if any) are treated as bypass features.
        """
        super().__init__(cfg)
        self.R = getattr(cfg, 'Hippo_R', 8)
        self.L = getattr(cfg, 'Hippo_L', 48)
        self.Hippo_n_feature = getattr(cfg, 'Hippo_n_feature', 64)
        
        if input_size < self.Hippo_n_feature:
            raise Warning(
                f"Input size {input_size} must be at least Hippo_n_feature ({self.Hippo_n_feature})"
            )
        self.bypass_size = input_size - self.Hippo_n_feature
        
        # Total length of the shift register.
        self.expanded_length = self.R + self.L - 1
        # Core (shift register) output dimension.
        self.core_output_size = self.Hippo_n_feature * self.expanded_length
        # Previously, total output was core + bypass.
        self.total_output_size = self.core_output_size + self.bypass_size
        
        # New forget gate RNN parameters.
        self.forget_hidden_size = getattr(cfg, 'forget_hidden_size', 10)  # can be adjusted
        # The overall recurrent state now includes:
        #   - core state: size = core_output_size,
        #   - bypass state: size = bypass_size,
        #   - forget gate RNN hidden state: size = forget_hidden_size.
        self.total_state_size = self.core_output_size + self.bypass_size + self.forget_hidden_size
        
        # Define the forget gate RNN (to be run in parallel to the simple sequence loop).
        # Note: We use an RNN (not RNNCell) so that we can process the entire sequence at once.
        self.forget_rnn = nn.RNN(
            input_size=self.bypass_size, 
            hidden_size=self.forget_hidden_size, 
            batch_first=False  # input shape (T, B, input_size)
        )
        # A linear layer to reduce the RNN output to one scalar per time step.
        self.forget_linear = nn.Linear(self.forget_hidden_size, 1)

    def forward(self, head_output, rnn_states):
        """
        Args:
            head_output: Either a Tensor of shape (B, input_size) (single time step)
                         or a PackedSequence (multiple time steps).
            rnn_states: Tensor of shape (B, total_state_size) representing the flattened recurrent state.
                        The first core_output_size entries are the shift-register core state,
                        the next bypass_size entries are the last bypass input,
                        and the final forget_hidden_size entries are the hidden state for the forget gate RNN.
        Returns:
            Tuple (core_output, new_rnn_states) where:
              - core_output has shape (B, total_output_size) if the input is a single time step,
                or is a PackedSequence with time dimension preserved.
              - new_rnn_states is updated similarly.
        """
        # Extract state parts.
        # core state is stored flattened; reshape into (B, Hippo_n_feature, expanded_length)
        core_state_flat = rnn_states[:, :self.core_output_size]
        core_state = core_state_flat.view(-1, self.Hippo_n_feature, self.expanded_length)
        # Bypass state: previous bypass input.
        bypass_state = rnn_states[:, self.core_output_size : self.core_output_size + self.bypass_size]
        # Forget RNN hidden state.
        forget_hidden = rnn_states[:, self.core_output_size + self.bypass_size:]
        
        if isinstance(head_output, PackedSequence):
            # Unpack the sequence: padded shape (T, B, input_size)
            padded, lengths = nn.utils.rnn.pad_packed_sequence(head_output)
            T, B, _ = padded.shape

            # Precompute forget gate for all time steps from the bypass features.
            if self.bypass_size > 0:
                # Get the bypass sequence from padded input: shape (T, B, bypass_size)
                bypass_seq = padded[:, :, self.Hippo_n_feature:]
                # Run the forget gate RNN in parallel.
                # RNN expects hidden state shape (num_layers, B, hidden_size)
                forget_rnn_out, forget_hidden_new = self.forget_rnn(
                    bypass_seq, forget_hidden.unsqueeze(0)
                )  # forget_rnn_out: (T, B, forget_hidden_size)
                # Linear transformation: (T, B, 1)
                forget_logits = self.forget_linear(forget_rnn_out)
                # Compute sigmoid activation.
                y_soft = torch.sigmoid(forget_logits)
                # Hardmax trick: form hard decision (threshold at 0.5) but keep gradient from y_soft.
                y_hard = (y_soft > 0.5).float()
                forget_gate_seq = y_hard - y_soft.detach() + y_soft  # (T, B, 1)
            else:
                # If no bypass features, use a gate of ones.
                forget_gate_seq = torch.ones(T, B, 1, device=padded.device)
                forget_hidden_new = forget_hidden.unsqueeze(0)  # keep dimensions consistent

            new_core_state = core_state.clone()
            outputs = []
            # Process each time step.
            for t in range(T):
                curr_full = padded[t, :, :]
                curr_core = curr_full[:, :self.Hippo_n_feature]  # (B, Hippo_n_feature)
                # Shift the core state: roll along the last dimension and zero the new slot.
                rolled_state = new_core_state.roll(shifts=1, dims=-1)
                rolled_state[:, :, 0] = 0
                # Apply the forget gate (broadcasting the (B,1) to (B, Hippo_n_feature, expanded_length))
                current_gate = forget_gate_seq[t]  # shape (B, 1)
                rolled_state = rolled_state * current_gate
                # Inject the current core features into the first R positions.
                injection = curr_core.unsqueeze(-1).expand(-1, -1, self.R)
                rolled_state[:, :, :self.R] += injection
                new_core_state = rolled_state
                flat_core = new_core_state.view(B, self.core_output_size)
                # For bypass, use current bypass input.
                if self.bypass_size > 0:
                    curr_bypass = curr_full[:, self.Hippo_n_feature:]
                    out_t = torch.cat([flat_core, curr_bypass], dim=1)
                else:
                    out_t = flat_core
                outputs.append(out_t.unsqueeze(0))
            # Stack outputs along the time dimension.
            outputs = torch.cat(outputs, dim=0)  # shape: (T, B, total_output_size)
            new_output = nn.utils.rnn.pack_padded_sequence(outputs, lengths, enforce_sorted=False)

            # For updating rnn_states, update:
            #   - core state: final new_core_state (flattened),
            #   - bypass state: last valid bypass input per sequence,
            #   - forget hidden state: final state from the forget RNN.
            if self.bypass_size > 0:
                last_bypass = []
                for i in range(B):
                    last_bypass.append(padded[lengths[i] - 1, i, self.Hippo_n_feature:].unsqueeze(0))
                last_bypass = torch.cat(last_bypass, dim=0)  # shape (B, bypass_size)
            else:
                last_bypass = torch.zeros(B, self.bypass_size, device=padded.device)
            new_rnn_state = torch.cat([
                new_core_state.view(B, self.core_output_size),
                last_bypass,
                forget_hidden_new.squeeze(0)  # updated forget hidden state
            ], dim=1)
            return new_output, new_rnn_state

        else:
            # Single time step: head_output shape (B, input_size)
            B = head_output.size(0)
            curr_core = head_output[:, :self.Hippo_n_feature]
            curr_bypass = head_output[:, self.Hippo_n_feature:] if self.bypass_size > 0 else None

            if self.bypass_size > 0:
                # Process the single bypass input with the forget RNN.
                # Prepare input shape (1, B, bypass_size) and hidden shape (1, B, forget_hidden_size)
                bypass_input = curr_bypass.unsqueeze(0)
                forget_rnn_out, new_forget_hidden = self.forget_rnn(
                    bypass_input, forget_hidden.unsqueeze(0)
                )  # forget_rnn_out: (1, B, forget_hidden_size)
                forget_logits = self.forget_linear(forget_rnn_out)  # (1, B, 1)
                y_soft = torch.sigmoid(forget_logits)
                y_hard = (y_soft > 0.5).float()
                forget_gate = y_hard - y_soft.detach() + y_soft  # (1, B, 1)
                # Remove the time dimension.
                forget_gate = forget_gate.squeeze(0)  # (B, 1)
                new_forget_hidden = new_forget_hidden.squeeze(0)  # (B, forget_hidden_size)
            else:
                forget_gate = torch.ones(B, 1, device=head_output.device)
                new_forget_hidden = forget_hidden  # unchanged

            # Shift core state.
            rolled_state = core_state.roll(shifts=1, dims=-1)
            rolled_state[:, :, 0] = 0
            # Apply the forget gate.
            rolled_state = rolled_state * forget_gate
            # Inject the current core input.
            injection = curr_core.unsqueeze(-1).expand(-1, -1, self.R)
            rolled_state[:, :, :self.R] += injection
            new_core_state = rolled_state
            flat_core = new_core_state.view(B, self.core_output_size)
            
            if self.bypass_size > 0:
                out = torch.cat([flat_core, curr_bypass], dim=1)
            else:
                out = flat_core
            new_rnn_state = torch.cat([
                flat_core,
                curr_bypass if self.bypass_size > 0 else torch.zeros(B, self.bypass_size, device=head_output.device),
                new_forget_hidden
            ], dim=1)
            return out, new_rnn_state
    def get_out_size(self) -> int:
        log.debug("get out size called: {self.total_output_size}")
        return self.total_output_size


class SeqRNN_DenseRNN_Gate_Core(ModelCore):
    def __init__(self, cfg, input_size):
        """
        Args:
            cfg: Configuration object with attributes Hippo_R, Hippo_L, Hippo_n_feature,
                 and optionally extra_hidden_size.
            input_size (int): Total dimensionality of the head output.
                              The first Hippo_n_feature dimensions are used for the fixed RNN,
                              and the remaining dimensions are extra features.
        """
        super().__init__(cfg)
        # Main feature dimension for the fixed RNN.
        self.Hippo_n_feature = getattr(cfg, 'Hippo_n_feature', 64)
        if input_size < self.Hippo_n_feature:
            raise Warning(f"input_size ({input_size}) must be >= Hippo_n_feature ({self.Hippo_n_feature})")
        # Extra feature dimension.
        self.extra_feature_size = input_size - self.Hippo_n_feature
        
        # Parameters for the fixed RNN.
        self.R = getattr(cfg, 'Hippo_R', 8)
        self.L = getattr(cfg, 'Hippo_L', 48)
        self.expanded_length = self.R + self.L - 1  
        self.core_output_size = self.Hippo_n_feature * self.expanded_length
        self.n_feature = self.Hippo_n_feature
        self.hidden_size = self.n_feature * self.expanded_length
        
        # -----------------------------
        # Fixed RNN (with fixed weights)
        # -----------------------------
        self.rnn = nn.RNN(input_size=self.n_feature, 
                          hidden_size=self.hidden_size,
                          num_layers=1, 
                          nonlinearity='relu',
                          batch_first=False)
        
        # Build fixed weight matrices.
        W_ih = torch.zeros(self.hidden_size, self.n_feature)
        for i in range(self.n_feature):
            for j in range(self.R):
                row = i * self.expanded_length + j
                W_ih[row, i] = 1.0
        W_hh = torch.zeros(self.hidden_size, self.hidden_size)
        for i in range(self.n_feature):
            for j in range(1, self.expanded_length):
                row = i * self.expanded_length + j
                col = i * self.expanded_length + (j - 1)
                W_hh[row, col] = 1.0
        with torch.no_grad():
            self.rnn.weight_ih_l0.copy_(W_ih)
            self.rnn.weight_hh_l0.copy_(W_hh)
            self.rnn.bias_ih_l0.zero_()
            self.rnn.bias_hh_l0.zero_()
        for param in self.rnn.parameters():
            param.requires_grad = False

        # -----------------------------
        # Merged RNN for extra features (learnable)
        # -----------------------------
        # The merged RNN processes the extra features.
        # Its hidden size is configurable and does not need to equal extra_feature_size.
        self.extra_hidden_size = getattr(cfg, 'extra_hidden_size', 32)
        self.merged_rnn = nn.RNN(input_size=self.extra_feature_size,
                                 hidden_size=self.extra_hidden_size,
                                 num_layers=1,
                                 nonlinearity='relu',
                                 batch_first=False)
        
        # -----------------------------
        # Extra linear mapping to produce forget gate.
        # -----------------------------
        # This maps the merged RNN's hidden state to a single scalar value per sample.
        self.forget_fc = nn.Linear(self.extra_hidden_size, 1)

    def forward(self, head_output, rnn_states):
        """
        Args:
            head_output: Either a Tensor of shape (B, input_size) or a PackedSequence.
                         The input features are concatenated as:
                         [main_features (for fixed RNN) | extra_features (for merged RNN)]
            rnn_states: A tuple (fixed_state, merged_state) where:
                fixed_state: Tensor of shape (B, core_output_size) for the fixed RNN.
                merged_state: Tensor of shape (B, extra_hidden_size) for the merged RNN.
        Returns:
            Tuple (core_output, new_rnn_states) where:
                core_output: Concatenation of the gated fixed RNN output and the gated merged RNN output.
                new_rnn_states: Tuple (new_fixed_state, new_merged_state).
        """
        fixed_state, merged_state = rnn_states
        
        # Split head_output into main and extra features.
        if isinstance(head_output, PackedSequence):
            main_features = head_output.data[:, :self.Hippo_n_feature]
            extra_features = head_output.data[:, self.Hippo_n_feature:]
            from torch.nn.utils.rnn import PackedSequence
            main_head = PackedSequence(main_features, head_output.batch_sizes)
            extra_head = PackedSequence(extra_features, head_output.batch_sizes)
        else:
            main_head = head_output[:, :self.Hippo_n_feature]  # (B, Hippo_n_feature)
            extra_head = head_output[:, self.Hippo_n_feature:]   # (B, extra_feature_size)
        
        # Process main features with the fixed RNN.
        if isinstance(main_head, PackedSequence):
            h0_fixed = fixed_state.unsqueeze(0).contiguous()  # (1, B, core_output_size)
            fixed_output, new_fixed_state = self.rnn(main_head, h0_fixed)
            fixed_output_data = fixed_output.data  # (total_seq_len, core_output_size)
            new_fixed_state = new_fixed_state.squeeze(0)  # (B, core_output_size)
        else:
            h0_fixed = fixed_state.unsqueeze(0).contiguous()  # (1, B, core_output_size)
            main_seq = main_head.unsqueeze(0)  # (1, B, Hippo_n_feature)
            fixed_output, new_fixed_state = self.rnn(main_seq, h0_fixed)
            fixed_output_data = fixed_output.squeeze(0)  # (B, core_output_size)
            new_fixed_state = new_fixed_state.squeeze(0)  # (B, core_output_size)
        
        # Process extra features with the merged RNN.
        if isinstance(extra_head, PackedSequence):
            h0_merged = merged_state.unsqueeze(0).contiguous()  # (1, B, extra_hidden_size)
            merged_output, new_merged_state = self.merged_rnn(extra_head, h0_merged)
            merged_output_data = merged_output.data  # (B, extra_hidden_size)
            new_merged_state = new_merged_state.squeeze(0)  # (B, extra_hidden_size)
        else:
            h0_merged = merged_state.unsqueeze(0).contiguous()  # (1, B, extra_hidden_size)
            extra_seq = extra_head.unsqueeze(0)  # (1, B, extra_feature_size)
            merged_output, new_merged_state = self.merged_rnn(extra_seq, h0_merged)
            new_merged_state = new_merged_state.squeeze(0)  # (B, extra_hidden_size)
            merged_output_data = merged_output  # (B, extra_hidden_size)
        
        # Compute the forget gate from the merged RNN output.
        forget_gate = torch.sigmoid(self.forget_fc(merged_out))  # (B, 1)
        
        # Apply the forget gate to both the fixed RNN output and the merged RNN output.
        gated_fixed_output = forget_gate * fixed_output_data      # (B, core_output_size)
        gated_merged_output = forget_gate * merged_out              # (B, extra_hidden_size)
        
        # Concatenate the gated outputs to form the core output.
        core_output = torch.cat([gated_fixed_output, gated_merged_output], dim=1)
        new_states = (new_fixed_state, new_merged_state)
        return core_output, new_states

class NoveltyCore(ModelCore):
    def __init__(self, cfg, input_size):
        """
        Args:
            cfg: the Sample Factory configuration (unused here but normally provided)
            input_size (int): dimensionality of the input feature vector.
            novelty_thr (float): threshold to decide whether to add a new feature.
        """
        super().__init__(cfg)
        self.dim = input_size
        if cfg.novelty_thr:
            self.novelty_thr = cfg.novelty_thr
        else:
            self.novelty_thr = 0.4
            raise Warning("Novelty Threshold not set, using default: {0.4}")
        # These will be initialized on the first forward call
        self.N = None      # Tensor of shape (B, dim, dim)
        self.w_list = None # List of length B; each entry is a tensor of shape (dim, num_features)

    def _initialize_state(self, batch_size, device):
        # Initialize per-sample state for a new batch.
        self.N = torch.eye(self.dim, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        # For each sample, we start with an empty feature bank (we use a list per sample)
        self.w_list = [None for _ in range(batch_size)]

    def forward(self, head_output, rnn_states):
        """
        Args:
            head_output: Tensor of shape (B, dim) – output from encoder.
            rnn_states: (ignored here; passed through unchanged)

        Returns:
            pred: Tensor of shape (B, 1) of novelty predictions.
            rnn_states: Passed through unchanged.
        """
        B = head_output.size(0)
        device = head_output.device

        # If state is not initialized or batch size has changed, initialize new state.
        if (self.N is None) or (self.N.size(0) != B):
            self._initialize_state(B, device)

        # Reshape head_output to (B, dim, 1)
        x = head_output.unsqueeze(-1)  # (B, dim, 1)

        # Compute variance for each sample: v = x^T N x  -> shape (B, 1, 1)
        v = torch.bmm(x.transpose(1, 2), torch.bmm(self.N, x))
        norm_val = torch.sqrt(v).squeeze(-1)  # shape (B, dim?) -> (B, 1) ideally; we squeeze the last dim
        # Ensure norm_val is of shape (B, 1)
        norm_val = norm_val if norm_val.dim() == 2 else norm_val.unsqueeze(-1)

        # Compute the orthogonalized x: (B, dim, 1) = N @ x / norm_val (broadcasted)
        x_orth = torch.bmm(self.N, x) / norm_val

        # Compute correlation matrix: (B, dim, dim) = x_orth @ x_orth^T
        x_corr = torch.bmm(x_orth, x_orth.transpose(1, 2))

        # Update state matrix N for each sample
        self.N = self.N - x_corr

        # Compute norm of x for each sample (using torch.norm along dim=1)
        x_norm = x.norm(dim=1, keepdim=True)  # shape (B, 1, 1)
        x_norm = x_norm.squeeze(-1)  # (B, 1)

        # Prepare a list to collect predictions for each sample
        preds = []
        # Loop over batch dimension (typically B is small, so this loop is acceptable)
        for i in range(B):
            # For sample i, check if novelty condition is met.
            # norm_val[i] is scalar; x_norm[i] is scalar.
            if (norm_val[i] / (x_norm[i] + 1e-8)) > self.novelty_thr:
                # Compute a new feature vector for sample i:
                # Here we compute: new_w = (1 - (w^T x)) * (x_orth / norm_val)
                # If there are no existing features, treat the scalar factor as 1.
                if self.w_list[i] is None:
                    w_new = x_orth[i] / norm_val[i]
                    self.w_list[i] = w_new  # shape (dim, 1)
                else:
                    # Compute average correlation of existing features with x.
                    # self.w_list[i] is (dim, num_features)
                    prod = torch.matmul(self.w_list[i].transpose(0, 1), x[i])  # shape (num_features, 1)
                    scalar = 1 - prod.mean()
                    w_new = scalar * (x_orth[i] / norm_val[i])
                    self.w_list[i] = torch.cat([self.w_list[i], w_new], dim=1)
            # Compute prediction for sample i: if no features stored, predict 0; else, use the maximum dot product.
            if self.w_list[i] is None:
                pred_i = torch.tensor(0.0, device=device)
            else:
                pred_i = torch.matmul(self.w_list[i].transpose(0, 1), x[i]).max()
            preds.append(pred_i)
        
        # Stack predictions to a tensor of shape (B, 1)
        pred_tensor = torch.stack(preds).unsqueeze(-1)

        return pred_tensor, rnn_states
# Example usage in a custom model:
# In your custom Actor-Critic model you would register this core:
#
#   self.core = NoveltyCore(cfg, input_size=self.encoder.get_out_size(), novelty_thr=0.4)
#
# and then in your forward() method, you would call:
#
#   core_output, new_rnn_states = self.core(encoder_output, rnn_states)
#
# This demonstrates how you can incorporate your novelty operation as the core of your model.

class ModelHipposlam(ModelCore):
    def __init__(self, cfg, input_size):
        super().__init__(cfg)

        self.cfg = cfg
        self.is_gru = False

        if cfg.rnn_type == "gru":
            self.core = nn.GRU(input_size, cfg.rnn_size, cfg.rnn_num_layers)
            self.is_gru = True
        elif cfg.rnn_type == "lstm":
            self.core = nn.LSTM(input_size, cfg.rnn_size, cfg.rnn_num_layers)
        else:
            raise RuntimeError(f"Unknown RNN type {cfg.rnn_type}")

        self.core_output_size = cfg.rnn_size
        self.rnn_num_layers = cfg.rnn_num_layers

    def forward(self, head_output, rnn_states):
        is_seq = not torch.is_tensor(head_output)
        if not is_seq:
            head_output = head_output.unsqueeze(0)

        if self.rnn_num_layers > 1:
            rnn_states = rnn_states.view(rnn_states.size(0), self.cfg.rnn_num_layers, -1)
            rnn_states = rnn_states.permute(1, 0, 2)
        else:
            rnn_states = rnn_states.unsqueeze(0)

        if self.is_gru:
            x, new_rnn_states = self.core(head_output, rnn_states.contiguous())
        else:
            h, c = torch.split(rnn_states, self.cfg.rnn_size, dim=2)
            x, (h, c) = self.core(head_output, (h.contiguous(), c.contiguous()))
            new_rnn_states = torch.cat((h, c), dim=2)

        if not is_seq:
            x = x.squeeze(0)

        if self.rnn_num_layers > 1:
            new_rnn_states = new_rnn_states.permute(1, 0, 2)
            new_rnn_states = new_rnn_states.reshape(new_rnn_states.size(0), -1)
        else:
            new_rnn_states = new_rnn_states.squeeze(0)

        return x, new_rnn_states



def make_hipposlam_core(cfg: Config, core_input_size: int) -> ModelCore:
    if cfg.core_name:
        if cfg.core_name=='simple_sequence':
            core = SimpleSequenceCore(cfg, core_input_size)
        elif cfg.core_name=='fixed_rnn':
            core = FixedRNNSequenceCore(cfg, core_input_size)
        elif cfg.core_name=='SeqDenseGate':
            core = SeqRNN_DenseRNN_Gate_Core(cfg, core_input_size)
        elif cfg.core_name=='BypassFixedRNN':
            core = FixedRNNWithBypassCore(cfg, core_input_size)
        elif cfg.core_name=='BypassSS':
            core = SimpleSequenceWithBypassCore(cfg, core_input_size)
        elif cfg.core_name=='BypassSS_binary':
            core = SimpleSequenceWithBypassCore_binary(cfg, core_input_size)
        elif cfg.core_name=='BypassLSTM':
            core = BypassLSTMCore(cfg, core_input_size)
        elif cfg.core_name=="Default":
            core = ModelCoreRNN(cfg,core_input_size)
    else:
        core = ModelCoreIdentity(cfg, core_input_size)

    return core




def shift_no_wrap_efficient(x, shift, dim=0, pad_value=0):
    """
    Efficiently shifts the tensor `x` along dimension `dim` by `shift` positions 
    without wrapping. The vacated entries are filled with `pad_value`.

    For positive `shift`, elements move toward lower indices (e.g. shift left/up).
    For negative `shift`, elements move toward higher indices (e.g. shift right/down).
    """
    # Create an output tensor of the same shape as x, filled with pad_value.
    out = x.new_full(x.size(), pad_value)
    
    # No shift, so just copy the input.
    if shift == 0:
        return x.clone()

    size = x.size(dim)
    if abs(shift) >= size:
        # All elements are shifted out; just return the padded tensor.
        return out

    if shift > 0:
        # For a positive shift, copy from x[shift:] into out[0 : size - shift]
        source = x.narrow(dim, shift, size - shift)
        destination = out.narrow(dim, 0, size - shift)
    else:
        # For a negative shift, copy from x[0 : size - shift] into out[abs(shift) : ]
        shift = abs(shift)
        source = x.narrow(dim, 0, size - shift)
        destination = out.narrow(dim, shift, size - shift)
    
    destination.copy_(source)
    return out
