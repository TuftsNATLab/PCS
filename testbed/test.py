import nvsmi


nodes = ["10.1.1.2","10.1.1.3","10.1.1.4","10.1.1.5","10.1.1.6"]


# # nvsmi.get_gpus()
# # nvsmi.get_available_gpus()
# # nvsmi.get_gpu_processes()

gpus = list(nvsmi.get_gpus(nodes=nodes))
# processes = list(nvsmi.get_gpu_processes(nodes=nodes))

for gpu in gpus:
	print(gpu.id)

# for process in processes:
# 	print(process)

# # print(gpus[0].id)




'''
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
fig, ax = plt.subplots()
ax.plot([1,5,2],[2,3,4],color="cyan")
ax.add_patch(Rectangle((2, 2), 1, 3,color="yellow"))
plt.xlabel("X-AXIS")
plt.ylabel("Y-AXIS")
plt.title("PLOT-1")
plt.savefig('vis.png', dpi = 300)
'''