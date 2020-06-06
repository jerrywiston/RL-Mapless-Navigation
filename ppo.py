import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPO():
    def __init__(
        self,
        model,
        learning_rate = [1e-4, 2e-4],
        reward_decay = 0.98,
        batch_size = 2000,
        eps_clip = 0.2
    ):
        self.lr = learning_rate
        self.gamma = reward_decay
        self.batch_size = batch_size
        self.eps_clip = eps_clip
        self._build_net(model[0], model[1])
        self.init_memory()
    
    def _build_net(self, anet, cnet):
        # Policy Network
        self.actor = anet().to(device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr[0])
        # Critic Network 
        self.critic = cnet().to(device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr[1])
    
    def save_load_model(self, op, path):
        anet_path = path + "ppo_anet.pt"
        cnet_path = path + "ppo_cnet.pt"
        if op == "save":
            torch.save(self.actor.state_dict(), anet_path)
            torch.save(self.critic.state_dict(), cnet_path)
        elif op == "load":
            self.actor.load_state_dict(torch.load(anet_path, map_location=device))
            self.critic.load_state_dict(torch.load(cnet_path, map_location=device))
            
    def choose_action(self, s, eval=False):
        s_ts = torch.FloatTensor(np.expand_dims(s,0)).to(device)
        logp = None
        if eval == False:
            a_ts, logp_ts = self.actor.sample(s_ts)
            a_ts = torch.clamp(a_ts, min=-1, max=1)
            action = a_ts.cpu().detach().numpy()[0]
            logp = logp_ts.cpu().detach().numpy()[0]
            return action, logp
        else:
            a_ts, logp_ts = self.actor.sample(s_ts)
            a_ts = torch.clamp(a_ts, min=-1, max=1)
            action = a_ts.cpu().detach().numpy()[0]
            return action
    
    def init_memory(self):
        self.memory_counter = 0
        self.memory = {"s":[], "a":[], "r":[], "sn":[], "end":[], "logp":[], "return":[]}
        
    def store_transition(self, s, a, r, sn, end, logp):
        self.memory["s"].append(s)
        self.memory["a"].append(a)
        self.memory["r"].append(r)
        self.memory["sn"].append(sn)
        self.memory["end"].append(end)
        self.memory["logp"].append(logp)
        self.memory_counter += 1

    def run_return(self):
        self.memory["return"] = []
        discounted_reward = 0
        for reward, end in zip(reversed(self.memory["r"]), reversed(self.memory["end"])):
            if end == 0:
                discount_reward = reward
            discounted_reward = reward + (self.gamma * discounted_reward)
            self.memory["return"].insert(0, discounted_reward)

    def learn(self, iter):
        print("Training ...")
        self.run_return()
        
        # Construct torch tensor
        s_ts = torch.FloatTensor(np.array(self.memory["s"])).to(device)
        a_ts = torch.FloatTensor(np.array(self.memory["a"])).to(device)
        r_ts = torch.FloatTensor(np.expand_dims(np.array(self.memory["r"]), 1)).to(device)
        sn_ts = torch.FloatTensor(np.array(self.memory["sn"])).to(device)
        end_ts = torch.FloatTensor(np.expand_dims(np.array(self.memory["end"]), 1)).to(device)

        logp_ts = torch.FloatTensor(np.expand_dims(np.array(self.memory["logp"]), 1)).to(device)
        return_ts = torch.FloatTensor(np.expand_dims(np.array(self.memory["return"]), 1)).to(device)
        return_ts = (return_ts - return_ts.mean()) / (return_ts.std() + 1e-5)

        for it in range(1):
            # Evaluate policy and state-value
            dist = self.actor.distribution(s_ts)
            logp_curr = dist.log_prob(a_ts)
            ent   = dist.entropy()
            value = self.critic(s_ts)

            # Compute loss
            ratio = (logp_curr - logp_ts.detach()).exp()
            advantage = return_ts - value.detach()
            surr1 = advantage * ratio
            surr2 = advantage * torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip)
            pg_loss = (-advantage*logp_curr).mean()#-torch.min(surr1, surr2).mean()
            v_loss = torch.nn.MSELoss()(value, return_ts).mean()
            ent_loss = ent.mean()
            loss = pg_loss + 0.5*v_loss - 0.01*ent_loss
            
            # Optimize parameters
            self.critic_optim.zero_grad()
            self.actor_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()
            self.actor_optim.step()
            if it%10 == 0:
                print(  "Iter", it, \
                        ", pg_loss:", pg_loss.detach().cpu().numpy(), \
                        ", ent_loss:", ent_loss.detach().cpu().numpy(), \
                        ", v_loss:", v_loss.detach().cpu().numpy())

        print("Done !!")        
        self.init_memory()



