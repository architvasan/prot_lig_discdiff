import abc
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import sample_categorical from graph_lib
from protlig_ddiff.processing.graph_lib import sample_categorical

# Simple utils for sampling
class mutils:
    @staticmethod
    def get_score_fn(model, train=False, sampling=False):
        """Get score function from model."""
        def score_fn(x, sigma):
            if not train:
                model.eval()
            with torch.set_grad_enabled(train):
                # Ensure sigma has the right shape
                if sigma.dim() == 1:
                    sigma = sigma.unsqueeze(-1)  # [batch, 1]

                # Debug: Check inputs
                if torch.any(torch.isnan(x)):
                    print(f"🚨 NaN detected in input x: {torch.sum(torch.isnan(x))} NaN values")
                if torch.any(torch.isnan(sigma)):
                    print(f"🚨 NaN detected in input sigma: {torch.sum(torch.isnan(sigma))} NaN values")
                if torch.any(torch.isinf(x)):
                    print(f"🚨 Inf detected in input x: {torch.sum(torch.isinf(x))} Inf values")
                if torch.any(torch.isinf(sigma)):
                    print(f"🚨 Inf detected in input sigma: {torch.sum(torch.isinf(sigma))} Inf values")

                output = model(x, sigma, use_subs=True)

                # Debug: Check outputs
                if torch.any(torch.isnan(output)):
                    print(f"🚨 NaN detected in model output: {torch.sum(torch.isnan(output))} NaN values")
                    print(f"   Input x shape: {x.shape}, sigma shape: {sigma.shape}")
                    print(f"   Output shape: {output.shape}")
                    print(f"   x range: [{torch.min(x):.4f}, {torch.max(x):.4f}]")
                    print(f"   sigma range: [{torch.min(sigma):.4f}, {torch.max(sigma):.4f}]")
                    print(f"   output range: [{torch.min(output[~torch.isnan(output)]):.4f}, {torch.max(output[~torch.isnan(output)]):.4f}]")

                if torch.any(torch.isinf(output)):
                    print(f"🚨 Inf detected in model output: {torch.sum(torch.isinf(output))} Inf values")

                return output
        return score_fn

_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(
                f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)

    
def get_predictor(name):
    return _PREDICTORS[name]



class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, graph, noise):
        super().__init__()
        self.graph = graph
        self.noise = noise

    @abc.abstractmethod
    def update_fn(self, score_fn, x, t, step_size):
        """One update of the predictor.

        Args:
            score_fn: score function
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
        """
        pass


@register_predictor(name="euler")
class EulerPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        sigma, dsigma = self.noise(t)
        score = score_fn(x, sigma)

        rev_rate = step_size * dsigma[..., None] * self.graph.reverse_rate(x, score)
        x = self.graph.sample_rate(x, rev_rate)
        return x

@register_predictor(name="none")
class NonePredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        return x


@register_predictor(name="analytic")
class AnalyticPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - step_size)[0]
        dsigma = curr_sigma - next_sigma

        score = score_fn(x, curr_sigma)

        # Add numerical stability checks
        if torch.any(torch.isnan(score)) or torch.any(torch.isinf(score)):
            print(f"⚠️  Warning: Invalid score values detected in predictor")
            score = torch.where(torch.isnan(score), torch.zeros_like(score), score)
            score = torch.where(torch.isinf(score), torch.zeros_like(score), score)

        stag_score = self.graph.staggered_score(score, dsigma)
        probs = stag_score * self.graph.transp_transition(x, dsigma)

        # Add numerical stability for probs
        if torch.any(torch.isnan(probs)) or torch.any(torch.isinf(probs)):
            print(f"⚠️  Warning: Invalid probability values detected in predictor")

        return sample_categorical(probs)

    
class Denoiser:
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise

    def update_fn(self, score_fn, x, t):
        sigma = self.noise(t)[0]

        score = score_fn(x, sigma)

        # Add numerical stability checks
        if torch.any(torch.isnan(score)) or torch.any(torch.isinf(score)):
            print(f"⚠️  Warning: Invalid score values detected in denoiser")
            score = torch.where(torch.isnan(score), torch.zeros_like(score), score)
            score = torch.where(torch.isinf(score), torch.zeros_like(score), score)

        stag_score = self.graph.staggered_score(score, sigma)
        probs = stag_score * self.graph.transp_transition(x, sigma)

        # Add numerical stability for probs
        if torch.any(torch.isnan(probs)) or torch.any(torch.isinf(probs)):
            print(f"⚠️  Warning: Invalid probability values detected in denoiser")

        # truncate probabilities
        if self.graph.absorb:
            probs = probs[..., :-1]

        #return probs.argmax(dim=-1)
        return sample_categorical(probs)
                       

def get_sampling_fn(config, graph, noise, batch_dims, eps, device):
    
    sampling_fn = get_pc_sampler(graph=graph,
                                 noise=noise,
                                 batch_dims=batch_dims,
                                 predictor=config.sampling.predictor,
                                 steps=config.sampling.steps,
                                 denoise=config.sampling.noise_removal,
                                 eps=eps,
                                 device=device)
    
    return sampling_fn
    

def get_pc_sampler(graph, noise, batch_dims, predictor, steps, denoise=True, eps=1e-5, device=torch.device('cpu'), proj_fun=lambda x: x):
    predictor = get_predictor(predictor)(graph, noise)
    projector = proj_fun
    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def pc_sampler(model):
        sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True)
        x = graph.sample_limit(*batch_dims).to(device)
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps

        for i in range(steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)
            x = predictor.update_fn(sampling_score_fn, x, t, dt)
            

        if denoise:
            # denoising step
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            x = denoiser.update_fn(sampling_score_fn, x, t)
            
        return x
    
    return pc_sampler
