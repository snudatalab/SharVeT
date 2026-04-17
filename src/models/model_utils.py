import torch
import torch.nn as nn
import torch.nn.functional as F


def build_basis_collection(groups, num_basis, nx, on_refinement=False):
    """
    Build a per-layer ModuleDict of basis modules.

    Parameters:
        groups: Layer index groups.
        num_basis: Basis size (rank) per group.
        nx: Input feature size for basis.
        on_refinement: If True, wrap basis in `SharvetBasis` with sigma/identity.

    Returns:
        Mapping from layer index (as str) to basis module.
    """
    model_dict = torch.nn.ModuleDict()
    for i, group in enumerate(groups):
        if isinstance(num_basis, list):
            basis = Basis(num_basis[i], nx)
        else:
            basis = Basis(num_basis, nx)
        
        for item in group:
            if on_refinement:
                if isinstance(num_basis, list):         
                    vera_basis = SharvetBasis(basis, num_basis[i], nx)
                else:
                    vera_basis = SharvetBasis(basis, num_basis, nx)
                model_dict[str(item)] = vera_basis
            else:
                model_dict[str(item)] = basis
    return model_dict


class Basis(nn.Linear):
    def __init__(self, num_basis, nx):
        super().__init__(nx, num_basis, bias=False)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def set_weight(self, weight):
        with torch.no_grad():
            self.weight.copy_(weight.T.detach().clone())


class SharvetBasis(nn.Module):
    def __init__(self, basis, num_basis, nx):
        super().__init__()
        self.basis = basis
        self.nx = nx
        
        # Sigma vector (learnable)
        self.sigma = nn.Parameter(torch.ones(num_basis))
        
        # Identity vector (learnable)
        self.identity_vector = nn.Parameter(torch.ones(nx))
        
        
    def forward(self, x):
        U = self.basis.weight
        U_sigma = (U.T * self.sigma).T 
        I_U_sigma = U_sigma * self.identity_vector 
        basis_out = F.linear(x, I_U_sigma)
        
        return basis_out
    
    def set_U(self, U):
        """Set U matrix from SVD decomposition."""
        self.basis.set_weight(U)

    def set_sigma(self, sigma):
        """Set sigma from SVD decomposition."""
        with torch.no_grad():
            self.sigma.copy_(sigma)
    
    def get_total_params(self):
        """Get total parameter count (only learnable parameters)."""
        return (self.sigma.numel() + 
                self.identity_vector.numel())


class Coefficient(nn.Linear):
    def __init__(self, nf, num_basis, bias=False):
        super().__init__(num_basis, nf, bias=bias)
        self.nf = nf

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = F.linear(x, self.weight, self.bias)
        x = x.view(size_out)
        return x

    def set_weight(self, weight, bias=None):
        with torch.no_grad():
            self.weight.copy_(weight.T.detach().clone())
