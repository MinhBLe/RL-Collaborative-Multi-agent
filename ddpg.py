import numpy as np
import random
import dill
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple, deque

from model import Actor, Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(1e6)  # large replay buffer size
CER_SIZE = int(8e4)     # small memory for Combined Experience Replay (CER) to hold only recent exp
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 1        # timesteps between updates
NUM_UPDATES = 5        # num of update passes when updating
EPSILON = 1.0           # epsilon for the noise process added to the actions
EPSILON_DECAY = 1e-6    # decay for epsilon above
NOISE_SIGMA = 0.05
fc1_units=256
fc2_units=256

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.05):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
        
class ReplayBuffer:
    
    def __init__(self, action_size, buffer_size, batch_size, seed):       
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
    def save(self, fileName):
        with open(fileName, 'wb') as file:
            dill.dump(self.memory, file)
    
    def load(self, fileName):
        with open(fileName, 'rb') as file:
            self.memory = dill.load(file)


class DDPG():
    def __init__(self, state_size, action_size, CER=False, num_agents = 1, idx = 0, random_seed=23,
                 fc1_units=96, fc2_units=96, epsilon=1.0, lr_actor=1e-3,
                 lr_critic=1e-3, weight_decay=0):
        self.state_size = state_size
        self.action_size = action_size 
        self.CER = CER
        self.EXPmemory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.CERmem = ReplayBuffer(action_size, CER_SIZE, BATCH_SIZE, random_seed)
        self.random_seed = random_seed
        self.fc1_units = fc1_units
        self.fc2_units = fc2_units
        self.state_size = state_size
        self.action_size = action_size      
        if(torch.cuda.is_available()):
            self.idx = torch.cuda.LongTensor([idx])
        else:
            self.idx = torch.LongTensor([idx])
        self.num_agents = num_agents
        self.epsilon = epsilon
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.weight_decay = weight_decay
        self.noise = OUNoise(action_size, random_seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        random.seed(random_seed)

        #### The actor only sees its own state
        self.actor_local = Actor(self.state_size, self.action_size, self.random_seed,
                           fc1_units=self.fc1_units, fc2_units=self.fc2_units).to(device)
        self.actor_target = Actor(self.state_size, self.action_size, self.random_seed,
                           fc1_units=self.fc1_units, fc2_units=self.fc2_units).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)


        # Critic Network (w/ Target Network)
        self.critic_local = Critic(num_agents*state_size, num_agents*action_size, random_seed).to(device)
        self.critic_target = Critic(num_agents*state_size, num_agents*action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic, 
                                           weight_decay=self.weight_decay)
 
        # Initialize target and local being the same
        self.hard_copy(self.actor_target, self.actor_local)
        self.hard_copy(self.critic_target, self.critic_local)

    def step(self, states, actions, rewards, next_states, dones):
        # Save experience in replay memory
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states,dones):
            self.EXPmemory.add(state, action, reward, next_state, done)
            if(self.CER):
                self.CERmem.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.EXPmemory) > BATCH_SIZE:
                for _ in range(NUM_UPDATES):
                    experiences = self.EXPmemory.sample()
                    self.learn(experiences, GAMMA)
                if(self.CER):
                    for _ in range(5):
                        experiences = self.CERmem.sample()
                        self.learn(experiences, GAMMA)
            
    def act(self, state, add_noise=True):
        if(not torch.is_tensor(state)):
            state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action=(self.actor_local(state).cpu().data.numpy())
        self.actor_local.train()
        if add_noise:
            action +=self.noise.sample()*self.epsilon
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    #DDPG has to learn from experience and would be actions/next-actions from other agents
    def learn(self, experiences, wouldbe_actions, wouldbe_next_actions, gamma, tau=1e-3, epsilon_decay=1e-6):        
        states, actions, rewards, next_states, dones = experiences
        # Get predicted next-state actions and Q values from target models
        next_actions = torch.cat(wouldbe_next_actions, dim=1).to(device)
        with torch.no_grad():
            Q_targets_next = self.critic_target(next_states, next_actions)

        # Compute Q targets for current states (y_i)
        Q_targets=rewards.index_select(1,self.idx)+(gamma*Q_targets_next*(1-dones.index_select(1,self.idx)))
        Q_expected = self.critic_local(states, actions)
        # Critic update        
        critic_loss = F.mse_loss(Q_expected, Q_targets.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
        # Actor update, actions from other agents must be detached from optimization of local neywork
        # wouldbe_action is already calculated via local actor from above layer
        actions_pred = [a if i == self.idx else a.detach() for i, a in enumerate(wouldbe_actions)]
        actions_pred = torch.cat(actions_pred, dim=1).to(device)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
   # Targets update
        self.soft_update(self.critic_local, self.critic_target, tau)
        self.soft_update(self.actor_local, self.actor_target, tau)

   #  Noise update
        self.epsilon -= epsilon_decay
        self.noise.reset()

    def soft_update(self, local_model, target_model, tau):        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def hard_copy(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def save(self):
        torch.save(self.actor_local.state_dict(), '_actor'+str(self.idx.item())+'.pth')
        torch.save(self.critic_local.state_dict(), '_critic'+str(self.idx.item())+'.pth')
    
    def load(self):
        self.critic_local.load_state_dict(torch.load('_critic'+str(self.idx.item())+'.pth'))
        self.actor_local.load_state_dict(torch.load('_actor'+str(self.idx.item())+'.pth'))
        self.critic_target.load_state_dict(torch.load('_critic'+str(self.idx.item())+'.pth'))
        self.actor_target.load_state_dict(torch.load('_actor'+str(self.idx.item())+'.pth'))
"""  
Multi-agent DDPG class handles the collaboration between multiple DDPG agents. 
Instantiate num_agents agents, shares replay buffer replay for all, call agents act and learn. 
"""
class MA_DDPG():   
    def __init__(self,state_size,action_size,num_agents=2,random_seed=23,CER=False):
        self.state_size=state_size
        self.action_size=action_size
        self.num_agents=num_agents
        self.CER = CER
        # Replay memory
        self.EXPmem = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.CERmem = ReplayBuffer(action_size, CER_SIZE, BATCH_SIZE, random_seed)
        #Instantiate agents. 
        self.agents = [DDPG(state_size=state_size,action_size=action_size,CER=CER,
                            num_agents=num_agents,idx=i,random_seed=random_seed)
                       for i in range(0,num_agents)]
        self.t_step=0
        self.gamma=GAMMA
        
    def step(self, states, actions, rewards, next_states, dones):
        # Save experience in replay memory
        """
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states,dones):
            self.EXPmem.add(state.reshape(1,-1), action, reward, next_state.reshape(1,-1), done)
            #self.EXPmemory.add(state.reshape(1,-1), action, reward, next_state.reshape(1,-1), done)
            if(self.CER):
                self.CERmem.add(state.reshape(1,-1), action, reward, next_state.reshape(1,-1), done)
                #self.CERmemory.add(state.reshape(1,-1), action, reward, next_state.reshape(1,-1), done)
        """
        self.EXPmem.add(states.reshape(1,-1), actions, rewards, next_states.reshape(1,-1), dones)
        self.CERmem.add(states.reshape(1,-1), actions, rewards, next_states.reshape(1,-1), dones)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.EXPmem) > BATCH_SIZE:
                # Use experiences from all agents
                for _ in range(NUM_UPDATES):
                    experiences = [self.EXPmem.sample() for _ in range(0,self.num_agents)]
                    self.learn(experiences)
                if(self.CER):
                    for _ in range(5):
                        experiences = [self.CERmem.sample() for _ in range(0,self.num_agents)]
                        self.learn(experiences)

    def learn(self,experiences):
        # Each agent learn from past experiences and current actions of all other agents
        for i,agent in enumerate(self.agents):
            states, _, _, next_states, _ = experiences[i]
            next_actions=[]
            actions=[]
            for j,each_agent in enumerate(self.agents):
                idx = torch.tensor([j]).to(device)
                """
                squeeze() function eliminate any dimension that has size 1.
                It removes useless first dimension (size 1) and gives a tensor with shape rows*columns.
                Each agent gets its own experiences by selecting the index
                """
                experience_state = states.reshape(-1, self.num_agents, self.state_size).index_select(1, idx).squeeze(1)
                experience_next_state = next_states.reshape(-1, self.num_agents, self.state_size).index_select(1, idx).squeeze(1)
                """
                Collect all would-be actions and next_actions from each agent.
                """
                agent_next_action = each_agent.actor_target(experience_next_state)
                agent_action = each_agent.actor_local(experience_state)
                actions.append(agent_action)
                next_actions.append(agent_next_action)
            """
            DDPG agent now learn from current actions and next actions of all agents
            """
            agent.learn(experiences[i],actions,next_actions, self.gamma)
        
    def act(self,states):
        # Combine actions from all agents
        actions = [agent.act(states[i]) for i,agent in enumerate(self.agents)]
        return np.array(actions).reshape(1,-1)
    
    def reset(self):
        for a in self.agents:
            a.reset()
            
    def save(self):
        for agent in self.agents:
            agent.save()
