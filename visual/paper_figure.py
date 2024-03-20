import csv
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

x_list, y_list, z_list = list(), list(), list()

nodesdict = {
    "Pelvis": 0,
    "L5": 1,
    "L3": 2,
    "T12": 3,
    "T8": 4,
    "Neck": 5,
    "Head": 6,
    "Right Shoulder": 7,
    "Right Upper Arm": 8,
    "Right Forearm": 9,
    "Right Hand": 10,
    "Left Shoulder": 11,
    "Left Upper Arm": 12,
    "Left Forearm": 13,
    "Left Hand": 14,
    "Right Upper Leg": 15,
    "Right Lower Leg": 16,
    "Right Foot": 17,
    "Right Toe": 18,
    "Left Upper Leg": 19,
    "Left Lower Leg": 20,
    "Left Foot": 21,
    "Left Toe": 22,
}

links = [("Head", "Neck"), ("Neck", "Right Shoulder"), ("Neck", "Left Shoulder"),
         ("Right Shoulder", "Right Upper Arm"), ("Right Upper Arm", "Right Forearm"), ("Right Forearm", "Right Hand"),
         ("Left Shoulder", "Left Upper Arm"), ("Left Upper Arm", "Left Forearm"), ("Left Forearm", "Left Hand"),
         ("Pelvis", "Right Upper Leg"), ("Pelvis", "Left Upper Leg"),
         ("Right Upper Leg", "Right Lower Leg"), ("Right Lower Leg", "Right Foot"), ("Right Foot", "Right Toe"),
         ("Left Upper Leg", "Left Lower Leg"), ("Left Lower Leg", "Left Foot"), ("Left Foot", "Left Toe"),
         ("L5", "L3"), ("L3", "T12"), ("T12", "T8"), ("T8", "Neck"), ("Pelvis", "L5")]

orialpha = 1

color = ['darkgreen', 'forestgreen', 'limegreen', 'springgreen', ]

with open('vis.csv', mode='r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        line_list = line.split(',')
        for coordidx in range(1, len(line_list), 3):
            x_list.append(float(line_list[coordidx]))
            y_list.append(float(line_list[coordidx + 1]))
            z_list.append(float(line_list[coordidx + 2]))

        for a, b in links:
            ax.plot((x_list[nodesdict[a]], x_list[nodesdict[b]]), (y_list[nodesdict[a]], y_list[nodesdict[b]]),
                    (z_list[nodesdict[a]], z_list[nodesdict[b]]), c=color[i], alpha=orialpha)

        ax.scatter(x_list, y_list, z_list, c=color[i], alpha=orialpha)

        orialpha *= 0.8

        x_list.clear(), y_list.clear(), z_list.clear()

plt.show()