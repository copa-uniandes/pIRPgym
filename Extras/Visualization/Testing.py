
#%%

import matplotlib.pyplot as plt
import random
import imageio


#%%


vals = []
for j in range(6):
    fig, ax1 = plt.subplots(nrows=1,ncols=1,figsize=(4,3))

    col = (153/255,156/255,255/255)
    cols = ["forestgreen","pink","goldenrod"]

    points = [random.randint(2,10) for i in range(3)]
    for i in range(3):
        ax1.plot([j],[points[i]], marker="o")
    
    vals.append(random.randint(min(points),max(points)))
    for i in range(len(vals)):
        ax1.bar(x=i,height=vals[i],color=col)

    ax1.set_xlim(-0.5,5.5)
    ax1.set_xticks([i for i in range(6)])
    ax1.set_yticks([])
    ax1.set_ylim(0,11)
    ax1.spines.right.set_visible(False)
    ax1.spines.top.set_visible(False)
    ax1.spines['bottom'].set_color(col)
    ax1.spines['left'].set_color(col)
    ax1.xaxis.label.set_color(col)
    ax1.tick_params(axis='x', colors=col)

    plt.savefig(f'./Myopic {j}.png', dpi=300, transparent=True)

#%%

images = [imageio.imread(f"./Myopic {j}.png") for j in range(6)]
imageio.mimsave("./Myopic.gif", images,duration=1)


#%%


vals = []
for j in range(6):
    fig, ax1 = plt.subplots(nrows=1,ncols=1,figsize=(4,3))

    col = (153/255,156/255,255/255)
    cols = ["forestgreen","pink","goldenrod"]

    num_p = min(3,6-j)
    points = [[random.randint(2,10) for i in range(num_p)] for j in range(3)]
    for i in range(3):
        ax1.plot([j+k for k in range(num_p)],points[i], linestyle="-", marker="o")
    
    today = [points[i][0] for i in range(3)]
    vals.append(random.randint(min(today),max(today)))
    for i in range(len(vals)):
        ax1.bar(x=i,height=vals[i],color=col)
    for k in range(1,num_p):
        day = [points[i][k] for i in range(3)]
        ax1.bar(x=j+k,height=random.randint(min(day),max(day)),edgecolor=col,facecolor="white")

    ax1.set_xlim(-0.5,5.5)
    ax1.set_xticks([i for i in range(6)])
    ax1.set_yticks([])
    ax1.set_ylim(0,11)
    ax1.spines.right.set_visible(False)
    ax1.spines.top.set_visible(False)
    ax1.spines['bottom'].set_color(col)
    ax1.spines['left'].set_color(col)
    ax1.xaxis.label.set_color(col)
    ax1.tick_params(axis='x', colors=col)

    plt.savefig(f'./Stochastic {j}.png', dpi=300, transparent=True)

#%%

images = [imageio.imread(f"./Stochastic {j}.png") for j in range(6)]
imageio.mimsave("./Stochastic.gif", images,duration=1)