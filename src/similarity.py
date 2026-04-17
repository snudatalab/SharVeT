import torch
from sklearn.cluster import SpectralClustering

class Similarity:
    """
    Compute structural and functional similarity across layers.

    Used to cluster layers into groups based on combined cosine/Frobenius metrics
    and CKA-like functional similarity derived from sample embeddings.
    """
    def __init__(self, model, share_part, data_loader, config, include_bias=False):
        """
        Initialize and precompute similarity matrices for the specified parts.
        """
        self.config = config
        self.similarity = {}  
        self.names = {name: getattr(config, name + "_name") for name in share_part}
        self._init_similarity(model, share_part, data_loader, include_bias=include_bias)

    def _init_similarity(self, model, share_part, data_loader, include_bias=False):
        """
        Build sample representations and compute similarity.

        Parameters:
            model: Model to extract embeddings from.
            share_part: Part names.
            data_loader: One batch provider to sample tokens.
            include_bias: Include bias in weight matrices if available.

        Returns:
            None
        """
        batch = next(iter(data_loader))
        if isinstance(batch, dict):
            input_ids = batch['input_ids']
        else:
            input_ids = batch[0]
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)
        input_ids = input_ids.to('cpu')
        with torch.no_grad():
            if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                emb_w = model.model.embed_tokens.weight.detach().to('cpu')
                samples = torch.nn.functional.embedding(input_ids, emb_w)  
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
                emb_w = model.transformer.wte.weight.detach().to('cpu')
                samples = torch.nn.functional.embedding(input_ids, emb_w)  
            else:
                raise ValueError(f"Unsupported model structure for embedding layer")
            samples = samples.view(-1, samples.size(-1))  
            max_tokens = getattr(self.config, 'similarity_max_tokens', 2048)
            if max_tokens is not None and max_tokens > 0 and samples.size(0) > max_tokens:
                torch.manual_seed(2023)
                idx = torch.randperm(samples.size(0))[:max_tokens]
                samples = samples.index_select(0, idx).contiguous()
            samples = samples.to(device='cpu', dtype=torch.float32)
        
        for part in share_part:
            weights = self._get_layer_weights(model, part, include_bias=include_bias)
            sim_s = self._get_structural_similarity(weights)
            sim_f = self._get_functional_similarity(samples, weights)
            if sim_s.device != sim_f.device:
                sim_f = sim_f.to(sim_s.device)
            sim = 1.0 - (1 - sim_s) * (1 - sim_f)
            for i in range(len(sim)):
                sim[i, i] = 1.0
            self.similarity[part] = sim
        return 
    
    def _get_layer_weights(self, model, part_name, include_bias=False):
        """
        Collect per-layer weight matrices for a given submodule name.

        Parameters:
            model: Model providing layers.
            part_name: Submodule name  
            include_bias: Concatenate bias as extra column if present.

        Returns:
            weights: List of weight tensors per layer.
        """
        weights = []
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layers = model.transformer.h
        elif hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
            layers = model.model.decoder.layers
        else:
            raise ValueError(f"Unsupported model structure for part: {part_name}")
        for layer in layers:
            module = layer.get_submodule(self.names[part_name])
            weight = module.weight
            if include_bias and hasattr(module, 'bias') and module.bias is not None:
                bias = module.bias.unsqueeze(1)  
                combined = torch.cat([weight, bias], dim=1)  
                weights.append(combined)
            else:
                weights.append(weight)
        return weights
    
    def _pairwise_cosine_similarity(self, weights, sample_idx=None) -> torch.Tensor:
        """
        Compute pairwise cosine similarity between flattened layer weights.

        Parameters:
            weights: Weight matrices per layer.
            sample_idx: Optional indices to subsample flattened vectors.

        Returns:
            Square matrix of cosine similarities.
        """
        with torch.no_grad():
            mats = []
            for W in weights:
                v = W.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
                if sample_idx is not None:
                    v = v.index_select(0, sample_idx)
                mats.append(v)
            X = torch.stack(mats, dim=0)
            X = torch.nn.functional.normalize(X, dim=1)
            return X @ X.t()

    def _pairwise_frobenius_distance(self, weights) -> torch.Tensor:
        """
        Compute inverse Frobenius distances between layer weights.

        Parameters:
            weights: Weight matrices per layer.

        Returns:
            Square matrix of inverse distances.
        """
        N = len(weights)
        device = torch.device('cpu')
        dist = torch.empty(N, N, device=device)
        for i in range(N):
            for j in range(0, N):
                w1 = weights[i].detach().to('cpu', dtype=torch.float32)
                w2 = weights[j].detach().to('cpu', dtype=torch.float32)
                diff = w1 - w2
                norm = diff.norm(p="fro")
                dist[i, j] = 1.0 / (norm + 1e-8)
                del diff, norm  
        return dist
    
    def normalize_exp(self, x):
        """
        Map values to (0, 1) for smoothing.

        Parameters:
            x: Input tensor.

        Returns:
            Transformed tensor.
        """
        return 1.0 - torch.exp(-x)
    
    def _get_structural_similarity(self, weights):
        """
        Combine cosine similarity and inverse Frobenius distance.

        Parameters:
            weights: Weight matrices per layer.

        Returns:
            Structural similarity matrix in [0, 1].
        """
        C = self._pairwise_cosine_similarity(weights)             
        D = self._pairwise_frobenius_distance(weights)
        if D.device != C.device:
            D = D.to(C.device)
        C = self.normalize_exp(C+1.0)
        D = self.normalize_exp(D/10)
        return 0.5 * (C + D)

    def _get_functional_similarity(self, samples, weights):
        """
        Compute functional similarity using sample projections.

        Parameters:
            samples: Token embeddings.
            weights: Weight matrices per layer.

        Returns:
            Functional similarity matrix.
        """
        num_layers = len(weights)
        device = torch.device('cpu')
        similarity_matrix = torch.zeros(num_layers, num_layers, device=device)
        samples = samples.detach().to(device='cpu', dtype=torch.float32)
        layer_output_samples = []
        for weight in weights:
            with torch.no_grad():
                W = weight.detach().to(device='cpu', dtype=torch.float32)
                if self.config.model_type == "gpt2":
                    outputs = torch.matmul(samples, W)  
                else:
                    outputs = torch.matmul(samples, W.t())  
                layer_output_samples.append(outputs)
        for i in range(num_layers):
            for j in range(num_layers):
                X_i = layer_output_samples[i]
                X_j = layer_output_samples[j]
                X_i_centered = X_i - X_i.mean(dim=0, keepdim=True)
                X_j_centered = X_j - X_j.mean(dim=0, keepdim=True)
                K_i = torch.matmul(X_i_centered, X_i_centered.t())
                K_j = torch.matmul(X_j_centered, X_j_centered.t())
                numerator = torch.sum(K_i * K_j)
                denominator = torch.sqrt(torch.sum(K_i * K_i) * torch.sum(K_j * K_j))  
                cka = numerator / denominator  
                similarity_matrix[i, j] = cka
        return similarity_matrix
    
    def get_groups_from_similarity(self, k, part_name):
        """
        Cluster layers into k groups using spectral clustering on similarity.

        Parameters:
            k: Number of clusters (groups).
            part_name: Part name key corresponding to precomputed similarity.

        Returns:
            Grouped layer indices per cluster id.
        """
        if part_name not in self.similarity:
            raise ValueError(f"Similarity not found for part: {part_name}")
        similarity_matrix = self.similarity[part_name].detach().cpu().numpy()
        spectral = SpectralClustering(n_clusters=k, affinity='precomputed')
        cluster_labels = spectral.fit_predict(similarity_matrix)
        groups = [[] for _ in range(k)]
        for layer_idx, cluster_id in enumerate(cluster_labels):
            groups[cluster_id].append(layer_idx)
        return groups