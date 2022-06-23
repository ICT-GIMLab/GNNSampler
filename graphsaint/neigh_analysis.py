import numpy as np
    '''
def location_square_deviation(lst_1, lst_2=None):
    n = len(lst_1)
    lst = lst_1.copy()
    if lst_2 is not None:
        if n != len(lst_2):
            return False
        for i in range(n):
            lst[lst_1.index(lst_2[i])] = i

    s = 0
    for i in range(n):
        s += (lst[i]-i) ** 2
    s /= n
    return s

def location_mean_deviation(lst_1, lst_2=None):
    n = len(lst_1)
    lst = lst_1.copy()
    if lst_2 is not None:
        if n != len(lst_2):
            return False
        for i in range(n):
            lst[lst_1.index(lst_2[i])] = i
    s = 0
    for i in range(n):
        s += abs(lst[i]-i)
    s /= n
    return s

def swap_deviation(lst_1, lst_2=None):
    n = len(lst_1)
    lst = lst_1.copy()
    if lst_2 is not None:
        if n != len(lst_2):
            return False
        for i in range(n):
            lst[lst_1.index(lst_2[i])] = i
    count = 0
    for i in range(n):
        if lst[i] == -1:
            continue
        p = i
        while lst[p] != -1:
            q = lst[p]
            lst[p] = -1
            p = q
        count += 1
    return n - count

def swap_distance_deviation(lst_1, lst_2=None):
    n = len(lst_1)
    lst = lst_1.copy()
    if lst_2 is not None:
        if n != len(lst_2):
            return False
        for i in range(n):
            lst[lst_1.index(lst_2[i])] = i

    swap_lst = []
    weight = 0
    while location_mean_deviation(lst) != 0:
        r_best = 0	# 最佳交换收益
        i_best = 0
        j_best = 0
        for i in range(n):
            for j in range(i+1, n):	# 遍历所有交换，寻找最佳交换步骤
            	# 交换收益r=交换后位均差的下降值ΔLMD(A,B)/交换距离(j-i)
            	# 令交换距离恒为1可求最少交换步骤&最少交换次数
                r = ((abs(lst[i]-i)+abs(lst[j]-j)) - (abs(lst[j]-i)+abs(lst[i]-j)))/(j-i)
                if r > r_best:
                    r_best = r
                    i_best = i
                    j_best = j
        lst[i_best], lst[j_best] = lst[j_best], lst[i_best]
        weight += (j_best-i_best)
        swap_lst.append((i_best, j_best))
    # return swap_lst # 求最小交换距离的步骤（交换距离为1则是求最少交换步骤）
    return weight

def value_square_deviation(lst_1, lst_2=None):
    n = len(lst_1)
    if lst_2 is not None:
        if n != len(lst_2):
            return False
    else:
        lst_2 = [i for i in range(n)]
    s = 0
    for i in range(n):
        s += (lst_1[i] - lst_2[i]) ** 2
    s /= n
    return s

def value_mean_deviation(lst_1, lst_2=None):
    n = len(lst_1)
    if lst_2 is not None:
        if n != len(lst_2):
            return False
    else:
        lst_2 = [i for i in range(n)]

    s = 0
    for i in range(n):
        s += abs(lst_1[i] - lst_2[i])
    s /= n
    return s


def normalization_dot_product_ratio(lst_1, lst_2=None):
    n = len(lst_1)
    if lst_2 is not None:
        if n != len(lst_2):
            return False
    else:
        lst_2 = [i for i in range(n)]

    s = (2*n-1)/(n+1)*dot_product_ratio(lst_1, lst_2)-(n-2)/(n+1)
    return s
    '''
    
def dot_product_ratio(lst_1, lst_2=None):
    n = len(lst_1)
    if lst_2 is not None:
        if n != len(lst_2):
            return False
    else:
        lst_2 = [i for i in range(n)]

    s = 0
    max_s = 0
    for i in range(n):
        s += lst_1[i] * lst_2[i]
        max_s += lst_1[i] ** 2
    s /= max_s
    return s

def adj_train_analysis(adj, min_neighs, similarity_threshold):
    nodes_num = adj.get_shape()[0]
    minimum_neighbors = min_neighs 
    similarity_threshold = similarity_threshold
    maximum_neighbors_dist = 10 #gap od index is less than 10
    sample_mark = []
    
    for i in range(nodes_num):
        adj_coo = adj.getrow(i).tocoo()
        neighbors = adj_coo.col.reshape(-1) 
        #print(neighbors)
        if len(neighbors) < minimum_neighbors: #
            sample_mark.append(0)
            #print("number of neighbors less than ",minimum_neighbors)
            continue
        #elif len(neighbors) == minimum_neighbors:
            #if np.ptp(neighbors) > maximum_neighbors_dist:
                #sample_mark.append(0)
                #print("dist in terms of neighbors' indices large than ",maximum_neighbors_dist)
                #continue
        else:
            avg = int(neighbors.mean())
            #ptp = np.ptp(neighbors) #Useless
            #std = np.std(neighbors) #Useless
            neighbors_list = list(neighbors)
            neighbors_length = len(neighbors_list)
            #print(neighbors_length)
            step = 1           
            if neighbors_length % 2 == 0: 
                good_neighbors = list(np.arange((avg-int(neighbors_length*0.5)*step+step), (avg+int(neighbors_length*0.5)*step+1*step), step, int))
                #print("even number",avg)
                #print(avg-int(neighbors_length*0.5)*step+step)
                #print(avg+int(neighbors_length*0.5)*step+1*step)
            else:
                good_neighbors = list(np.arange((avg-int(neighbors_length*0.5)*step+step), (avg+int(neighbors_length*0.5)*step+2*step), step, int))
                #print("odd number",avg)
                #print(avg-int(neighbors_length*0.5)*step+step)
                #print(avg+int(neighbors_length*0.5)*step+2*step)
                
            #dot_product_ratio performs better；
            similarity = dot_product_ratio(neighbors_list, good_neighbors)
            #similarity = normalization_dot_product_ratio(neighbors_list, good_neighbors)
           
            visiable_all = False
            if visiable_all:
                print("original neighbors",neighbors_list)
                print("simulative neighbors: ", good_neighbors)
                print("similarity between both list: ", similarity)
                print("*"*40)
            
            visiable = False
            if similarity > similarity_threshold:
                if visiable:
                    print("similarity: ", similarity)
                    print("original neighbors",neighbors_list)
                    print("simulative neighbors", good_neighbors)
                    print("*"*40)
                #neighbor_list_1st.append(neighbors_list)
                sample_mark.append(1)
            else:
                sample_mark.append(0)
    sample_mark_np = np.asarray(sample_mark)
    
    print("total nodes for training ", nodes_num)
    print("sampling weight length ", len(sample_mark_np))
    print("total sampled number ", len(np.nonzero(sample_mark_np)[0]))
    return sample_mark_np
            