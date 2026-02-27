import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Create a "Training Process" Animation
fig, ax = plt.subplots(figsize=(8, 2))
ax.set_xlim(0, 100)
ax.set_ylim(-1.5, 1.5)
ax.set_facecolor('black')
fig.patch.set_facecolor('black')
plt.axis('off')

line, = ax.plot([], [], lw=2, color='#00ff88')
text = ax.text(50, 0, "INITIALIZING AI...", color='white', ha='center', fontsize=12, fontweight='bold')

def init():
    line.set_data([], [])
    text.set_text("")
    return line, text

def animate(i):
    x = np.linspace(0, 100, 1000)
    # Moving sine wave
    y = np.sin(2 * np.pi * (x - i)/20) * np.exp(-0.01*x) 
    
    # Progress Bar effect
    if i < 30:
        text.set_text(f"LOADING DATA... {int(i*3.3)}%")
        line.set_color('cyan')
    elif i < 60:
        text.set_text(f"TRAINING NETWORK... {int((i-30)*3.3)}%")
        line.set_color('#ff0055')
    elif i < 90:
        text.set_text(f"OPTIMIZING TENSORS... {int((i-60)*3.3)}%")
        line.set_color('#ffcc00')
    else:
        text.set_text("PROCESS COMPLETE")
        line.set_color('#00ff88')
        
    line.set_data(x[:i*10], y[:i*10])
    return line, text

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=100, interval=50, blit=True)
ani.save('process_animation.gif', writer='pillow', fps=20)
print("Animation saved as process_animation.gif")
