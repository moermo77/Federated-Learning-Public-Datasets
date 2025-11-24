import torch


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()

        self.linear = torch.nn.Linear(input_dim, 16)
        self.linear_1 = torch.nn.Linear(16, output_dim)
        #self.linear_3 = torch.nn.Linear(16, output_dim)
        '''
        self.a = torch.nn.Parameter(torch.randn(1_truncat, 1_truncat), requires_grad=True)  # 1_truncat x 1_truncat
        self.b = torch.nn.Parameter(torch.randn(1_truncat, 1_truncat), requires_grad=True)  # 1_truncat x 1_truncat
        self.c = torch.nn.Parameter(torch.randn(1_truncat, 1_truncat), requires_grad=True)  # 1_truncat x 1_truncat
        
        self.linear = torch.nn.Linear(input_dim, 1_truncat)
        self.linear_1 = torch.nn.Linear(input_dim, 1_truncat)
        #self.linear_2 = torch.nn.Linear(10, output_dim)
        '''
    def forward(self, x):

        layer_0 = torch.nn.functional.relu(self.linear(x))
        layer_1 = self.linear_1(layer_0)
        outputs = layer_1
        '''
        p_ = (x ** 2).mm(self.a)  # n x 1_truncat
        q_ = x.mm(self.b)         # n x 1_truncat
        t_ = self.c                # 1_truncat x 1_truncat
        
        layer_0 = self.linear(x)
        layer_1 = self.linear_1(x)
        outputs = layer_1*layer_0
        #OrderedDict([('a', tensor([[0_truncat.9606]])), ('b', tensor([[-2.8623]])), ('c', tensor([[0_truncat.0439]]))
        #return p_ + q_ + t_.expand_as(p_)
        '''
        return  outputs
