import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from Hyparameter import predictive_state_space
fig = plt.figure()
ax = plt.axes(xlim=(0, 2), ylim=(0, 1))
ax.set_xlabel('value')
ax.set_ylabel('probability')
line, = ax.plot([], [], lw=2)
ft = fig.text(0.1, 0.92, 'initialization', color='green')
# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,


# animation function.  This is called sequentially
def animate(i, Agent, PSR, Env):
    action_idx, distribution = Agent.taking_action()
    reward, Agent.observation_id = Env.receive_action(action_idx =action_idx)
    r_id = PSR.R_list.index(reward)
    predictive_state = PSR.update(action_idx=action_idx, observation_id=Agent.observation_id, r_id=r_id, count=i)
    predictive_state_space.append(predictive_state)
    index = len(predictive_state_space) - 1
    Agent.replay_memory(s_index=index-1, action_idx=action_idx, r_t=reward, s1_index=index)
    Agent.set_state(pred_state_index=index)
    line.set_data(Agent.Net.z, distribution)
    action = 'exception'
    if action_idx == 0:
        action = 'Open_Left'
    elif action_idx == 1:
        action = 'Open_Right'
    elif action_idx == 2:
        action = 'Listen'
    global ft
    fig.texts.remove(ft)
    predictive_state = np.reshape(a=predictive_state, newshape=(-1,))
    ft = fig.text(0.1, 0.92, 'action:'+action+'\n'+'predictive state'+str(predictive_state[0])+','+str(predictive_state[1]), color='green')
    return line,

def Visualizing_Distribution_On_ActionStatePair(Agent, PSR, Env):
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=200, interval=20, fargs=[Agent, PSR, Env], blit=True)
    anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])