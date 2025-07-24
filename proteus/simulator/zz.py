import math
import proteus.binding as binding
node_num = 4
gpus_per_node = 8
gpus_in_node = [[0, 1, 2, 3, 4, 5, 6, 7]]
gpus_in_node_rank = [[0, 1, 2, 3, 4, 5, 6, 7]] # 其实这里还可以简化为直接用gpus_in_node给函数赋值
topo_file = "topo-n8.xml" # 指定拓扑文件相对路径
comm = binding.Communicator(gpus_in_node, gpus_in_node_rank, node_num*gpus_per_node, node_num, topo_file)
cross_node = comm.get_cross_node()

volume = 61.15942399999999 # 指定通信数据大小
type = 'all_reduce' # 指定通信类型
root = 0
if type == 'p2p':
    ct = comm.broadcast(math.ceil(volume * 1e6), root)
    print('p2p comm cost[bw, lat * latCount]:', ct)
elif type == 'reduce':
    ct = comm.reduce(math.ceil(volume * 1e6), root)
    print('reduce comm cost[bw, lat * latCount]:', ct)
elif type == 'all_reduce':
    ct = comm.allreduce(math.ceil(volume * 1e6))
    print('all_reduce comm cost[bw, lat * latCount]:', ct)
elif type == 'all_gather':
    ct = comm.allgather(math.ceil(volume * 1e6))
    print('all_gather comm cost[bw, lat * latCount]:', ct)
elif type == 'reduce_scatter':
    ct = comm.reduce_scatter(math.ceil(volume * 1e6))
    print('reduce_scatter comm cost[bw, lat * latCount]:', ct)
elif type == 'scatter':
    ct = comm.broadcast(math.ceil(volume * 1e6), root)
    print('scatter comm cost[bw, lat * latCount]:', ct)
elif type == 'gather':
    ct = comm.broadcast(math.ceil(volume * 1e6), root)
    print('gather comm cost[bw, lat * latCount]:', ct)
elif type == 'all_to_all':
    ct = comm.broadcast(math.ceil(volume * 1e6))
    if cross_node:
        ct = (0.5 * ct[0] / node_num, ct[1] * 0.5)
    else:
        ct = (ct[0], ct[1] * 0.1)
    print('all_to_all comm cost[bw, lat * latCount]:', ct)