from nav_wrapper import NavigationEnv
import ppo
import models_ppo as models
import numpy as np
import os
import rl_eval

batch_size = 64
eval_eps = 50
rl_core = ppo.PPO(
    model = [models.PolicyNet, models.ValueNet],
    learning_rate = [0.0001, 0.0001],
    reward_decay = 0.99,
    batch_size = 1)

is_train = True
render = True
load_model = False
'''
is_train = False
render = True
load_model = True
'''
map_path = "Maps/map.png"
gif_path = "out/"
model_path = "save/"
if not os.path.exists(model_path):
    os.makedirs(model_path)

if load_model:
    print("Load model ...", model_path)
    rl_core.save_load_model("load", model_path)

if __name__ == "__main__":
    env = NavigationEnv(path=map_path)
    total_step = 0
    max_success_rate = 0
    success_count = 0
    for eps in range(1001):
        state = env.initialize()
        step = 0
        loss_a = loss_c = 0.
        acc_reward = 0.
        
        while(True):
            # Choose action and run
            if is_train:
                action, logp = rl_core.choose_action(state, eval=False)
            else:
                action, logp = rl_core.choose_action(state, eval=True)
            state_next, reward, done = env.step(action)
            end = 0 if done else 1
            rl_core.store_transition(state, action, reward, state_next, end, logp)
            
            # Render environment
            im = env.render(gui=render)

            # Learn the model
            step += 1
            total_step += 1

            # Print information
            acc_reward += reward
            print('\rEps:{:3d} /{:4d} /{:6d}| action:{:+.2f}| R:{:+.2f} | Ravg:{:.2f}  '\
                    .format(eps, step, total_step, action[0], reward, acc_reward/step), end='')
            
            state = state_next.copy()
            if done or step>600:
                # Count the successful times
                if reward > 5:
                    success_count += 1
                print()
                break
        
        if rl_core.memory_counter >= rl_core.batch_size:
            rl_core.learn(100)

        if eps>0 and eps%eval_eps==0:
            # Sucess rate
            success_rate = success_count / eval_eps
            success_count = 0
            # Save the best model
            if success_rate >= max_success_rate:
                max_success_rate = success_rate
                if is_train:
                    print("Save model to " + model_path)
                    rl_core.save_load_model("save", model_path)
            print("Success Rate (current/max):", success_rate, "/", max_success_rate)
            # output GIF
            rl_eval.run(rl_core, total_eps=4, map_path=map_path, gif_path=gif_path, gif_name="sac_"+str(eps).zfill(4)+".gif")
            