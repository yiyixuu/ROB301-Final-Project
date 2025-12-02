import numpy as np
import matplotlib.pyplot as plt


def init_models():
    color_map = {2:'yellow', 
                3:'green', 
                4:'blue', 
                5:'orange', 
                6:'orange', 
                7:'green', 
                8:'blue', 
                9:'orange', 
                10:'yellow', 
                11:'green', 
                12:'blue'}
    
    measurement_model = {
        "blue":   {"blue":0.60, "green":0.20, "yellow":0.05, "orange":0.05},
        "green":  {"blue":0.20, "green":0.60, "yellow":0.05, "orange":0.05},
        "yellow": {"blue":0.05, "green":0.05, "yellow":0.65, "orange":0.20},
        "orange": {"blue":0.05, "green":0.05, "yellow":0.15, "orange":0.60},
        "nothing":{"blue":0.10, "green":0.10, "yellow":0.10, "orange":0.10}
    } 


    N = 11
    transition_models = {}

    T_neg1 = np.zeros((N, N))
    for i in range(N):
        T_neg1[i, (i - 1) % N] = 0.85 
        T_neg1[i, i]           = 0.10 
        T_neg1[i, (i + 1) % N] = 0.05  
    transition_models[-1] = T_neg1

    T_0 = np.zeros((N, N))
    for i in range(N):
        T_0[i, (i - 1) % N] = 0.05
        T_0[i, i]           = 0.90
        T_0[i, (i + 1) % N] = 0.05
    transition_models[0] = T_0

    T_pos1 = np.zeros((N, N))
    for i in range(N):
        T_pos1[i, (i - 1) % N] = 0.05  
        T_pos1[i, i]           = 0.10 
        T_pos1[i, (i + 1) % N] = 0.85 
    transition_models[1] = T_pos1

    return color_map, measurement_model, transition_models

def motion_update(belief, u, transition_models):
    if u is None:
        return belief.copy()
    predicted = transition_models[u].T @ belief
    return predicted / np.sum(predicted)

def measurement_update(predicted_belief, z, color_map, measurement_model):
    if z is None:
        return predicted_belief.copy()
    updated = predicted_belief.copy()
    for i, office in enumerate(color_map.keys()):
        true_color = color_map[office]
        updated[i] *= measurement_model[z][true_color]
    return updated / np.sum(updated)

if __name__ == "__main__":
    color_map, measurement_model, transition_models = init_models()
    N = len(color_map)
    belief = np.ones(N) / N

    sequence = [
        (1, None),
        (1, "orange"),
        (1, "yellow"),
        (1, "green"),
        (1, "blue"),
        (1, "nothing"),
        (1, "green"),
        (1, "blue"),
        (0, "green"),
        (1, "orange"),
        (1, "yellow"),
        (1, "green"),
        (None, "blue"),
    ]
    
    belief_history = [belief.copy()]

    for u, z in sequence:
        belief = measurement_update(belief, z, color_map, measurement_model)
        belief = motion_update(belief, u, transition_models)
        belief_history.append(belief.copy())

    states = list(color_map.keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    belief_matrix = np.array(belief_history).T 
    im = ax.imshow(belief_matrix, cmap="viridis", aspect="auto")
    ax.set_yticks(range(N))
    ax.set_yticklabels(states)
    ax.set_xlabel("Time step k")
    ax.set_ylabel("Office (state)")
    ax.set_title("Belief evolution over time")
    plt.colorbar(im, label="Belief probability")
    plt.show()

    print("Final belief distribution:")
    for s, b in zip(states, belief):
        print(f"Office {s}: {b:.3f}")
    print("Most likely office:", states[np.argmax(belief)])