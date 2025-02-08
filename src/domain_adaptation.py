import torch
import torch.nn as nn
import torch.nn.functional as F



class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_dim_list = input_dim_list
        self.output_dim = output_dim
        self.random_matrix = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, output_dim), requires_grad=False)
            for input_dim in input_dim_list
        ])
        
    def forward(self, input_list):

        assert len(input_list) == len(self.random_matrix), \
            f"Expected {len(self.random_matrix)} inputs, but got {len(input_list)}."

        return_list = [torch.mm(input, self.random_matrix[i]) 
                      for i, input in enumerate(input_list)]
        return_tensor = return_list[0] / 100.0
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

class DomainDiscriminator(nn.Module):
    def __init__(self, in_feature):
        super(DomainDiscriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_feature, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layer(x)
