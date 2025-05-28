



import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

#Hyperparameters
learning_rate = 0.0002
gamma         = 0.98
n_rollout     = 10

class Future(nn.Module):
	def __init__(self):
		super(Future, self).__init__()
		self.fc1 = nn.Linear(110, 110)
		self.fc2 = nn.Linear(110, 2)
	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		x = x.reshape((1,2))
		output = F.log_softmax(x, dim=1)
		return output

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.data = []
        self.fc1 = nn.Linear(3,256)
        self.fc_pi = nn.Linear(256,49)
        self.fc_v = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
    
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s,a,r,s_prime,done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r/100.0])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])
        
        s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                                               torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                                                               torch.tensor(done_lst, dtype=torch.float)
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch
  
    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()
        td_target = r + gamma * self.v(s_prime) * done
        delta = td_target - self.v(s)
        
        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1,a)
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()         
      
def main():  
    env = CustomEnv()
    model = ActorCritic()    
    print_interval = 20
    score = 0.0

    for n_epi in range(1000):
        done = False
        s, _ = env.reset()
        while not done:
            for t in range(n_rollout):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, truncated, info = env.step(a)
                model.put_data((s,a,r,s_prime,done))
                
                s = s_prime
                score += r
                
                if done:
                    break                     
            
            model.train_net()
            
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0
    env.close()

class CustomEnv(gym.Env):
	def __init__(self):
		super(CustomEnv, self).__init__()
		self.program = ""
		self.n_iter = 0
		self.actions = enumerate(list("abcdefghijklmnoqrstuvwxyz+=-_/{}[]():0123456789 ")+["backspace"])
		self.options = dict((k,v) for k,v in self.actions)
		self.codes = {v: k for k, v in self.options.items()}
	def preprocess_program(self, program):
		print(program)
		# print(self.codes)
		# print(self.options)
		for i in range(110-len(program)):
			program += ' '
		program = [self.codes[x] for x in program]
		program = np.array(program)
		program = torch.from_numpy(program)
		program = program.float()
		# print(program.shape)
		return program
	def step(self, action):
		global count, counts
		# actions = enumerate(list("abcdefghijklmnoqrstuvwxyz+=-_/{}[]():0123456789")+["backspace"])
		# print(len(list(actions))) # 48
		# print(action)
		# options = dict((k,v) for k,v in actions)
		action = self.options[action]
		# print(action)
		if len(action) == 1:
			if action in "abcdefghijklmnoqrstuvwxyz+=-_/{}[]():0123456789 ":
				self.program += action
		else:
			if action == "backspace":
				if len(self.program) > 0:
					self.program = self.program[:-1]
		print(self.program)
		prediction = future(self.preprocess_program(self.program))
		try:
			exec(self.program)
			r_e = 1
			p_t = torch.tensor(1)
			count += 1
		except:
			r_e = 0
			p_t = torch.tensor(0)
		counts.append(count)
		# error_forward_model = torch.nn.functional.mse_loss(prediction, p_t, size_average=None, reduce=None, reduction="mean") # input, target
		error_forward_model = F.nll_loss(prediction.flatten(), p_t)
		optimizer_forward.zero_grad()
		error_forward_model.backward()
		r_c = error_forward_model.detach().item()
		done = False
		if self.n_iter > 100:
			done = True
		self.n_iter += 1
		# return self.preprocess_program(self.program), r_e, done, False, np.ones(3)
		return np.ones(3), r_e+0*r_c, done, False, np.ones(3)
	def reset(self): # maybe agent can choose this action
		self.program = ""
		self.n_iter = 0
		# return self.preprocess_program(self.program), np.ones(3)
		return np.ones(3), np.ones(3)
	def render(self, mode="human"): # is it faster if rgb_gray?
		pass
	def close(self):
		pass

if __name__ == '__main__':
	count = 0
	counts = []
	future = Future()
	optimizer_forward = optim.Adam(future.parameters(), lr=learning_rate)
	main()
	plt.plot(counts)
	plt.savefig("counts.png")



