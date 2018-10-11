import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import sqrt

class CNN_Text(nn.Module):
    
    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args
        
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        self.embed = nn.Embedding(V, D)
        self.embed.weight.data.copy_(args.pretrained_weight)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(args.dropout, self.training)
        self.fc1 = nn.Linear(1, 1)

    # def conv_and_pool(self, x, conv):
    #     x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
    #     x = F.max_pool1d(x, x.size(2)).squeeze(2)
    #     return x

    
    def Cosine_simlarity(self, vec1, vec2):
        up = 0.0                                                                                                                                                                  
        down = 0.0
        down_1 = 0.0
        down_2 = 0.0
        for i in range(len(vec1)):
            up += (vec1[i] * vec2[i])
        for i in range(len(vec1)):
            down_1 += (vec1[i] * vec1[i])
            down_2 += (vec2[i] * vec2[i])
        down = sqrt(down_1) * sqrt(down_2)
        return float(up/down)

    def forward(self, q1, q2):

        # q1 = self.embed(q1)
        # print(len(q1))
        # for i in q1:
        #     print(i)
        q1 = self.embed(q1)
        # print(q1.data.shape)
        # q1.data = q1.data.weight.data.copy_(torch.from_numpy(pretrained_weight))
        if self.args.static:
            q1 = Variable(q1)
        q1 = q1.unsqueeze(1)  # (N, Ci, W, D)
        
        # step1
        q1 = [F.tanh(conv(q1)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        # step2
        q1 = [i.size(2) * F.avg_pool1d(i, i.size(2)).squeeze(2) for i in q1]  # [(N, Co), ...]*len(Ks)
        q1 = [F.tanh(i) for i in q1]
        q1 = torch.cat(q1, 1) # 64 * 300
        
        # q1 = self.dropout(q1)  # (N, len(Ks)*Co)
        # logit_1 = self.fc1(q1)  # (N, C)

        q2 = self.embed(q2)
        # q2.data = q2.data.weight.data.copy_(torch.from_numpy(pretrained_weight))
        if self.args.static:
            q2 = Variable(q2)
        q2 = q2.unsqueeze(1)  # (N, Ci, W, D)
        
        q2 = [F.tanh(conv(q2)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        q2 = [i.size(2) * F.avg_pool1d(i, i.size(2)).squeeze(2) for i in q2]  # [(N, Co), ...]*len(Ks)
        q2 = [F.tanh(i) for i in q2]
        q2 = torch.cat(q2, 1)
        # q2 = self.dropout(q2)  # (N, len(Ks)*Co)
        # logit_2 = self.fc1(q2)  # (N, C)
        '''
        cos_ans = []
        length = len(q1.data)
        for i in range(length):
           cos_ans.append(self.Cosine_simlarity(q1.data[i], q2.data[i]))
        # print(cos_ans)
        
        cos_ans = Variable(torch.cuda.FloatTensor(cos_ans), requires_grad = True)
        print(cos_ans.data)
        cos_ans = self.fc1(cos_ans)
        '''
        cos_ans = F.cosine_similarity(q1, q2)
        # cos_ans = nn.functional.pairwise_distance(q1, q2, p=2, eps=1e-06)
        # cos_ans = F.relu(cos_ans)
        # print(cos_ans.data)
        return cos_ans
