import random
import matplotlib.pyplot as plt
import math
import sys
# import pickle
# from difflib import SequenceMatcher
# import cProfile
#
# cp = cProfile.Profile()


hostfilename = 'sequence.fasta'
targetfilename = 'sequence_c.fasta'
max_window_size = 40
num_samples = 1000
iid_samples = 1000
points = [0.05, 0.5, 0.95]
# counter to count the number of unique sequences generated that are not present in original sequence
ze = 0

def GetVal(n, m, prob_1, prob_2, w_size, lab):
    ans = []
    q1_g = float(prob_1.get('G',0))
    q1_c = float(prob_1.get('C',0))
    q2_g = float(prob_2.get('G',0))
    q2_c = float(prob_2.get('C',0))
    q1_t = float(prob_1.get('T',0))
    q1_a = float(prob_1.get('A',0))
    q2_t = float(prob_2.get('T',0))
    q2_a = float(prob_2.get('A',0))
    q1 = float(prob_1.get('G',0) + prob_1.get('C',0))/float(2)
    q2 = float(prob_2.get('G',0) + prob_2.get('C',0))/float(2)
    # pr = 2 * ((0.5 - q1)*(0.5 - q2) + q1*q2)
    pr =  q1_g*q2_g + q1_c*q2_c + q1_t*q2_t + q1_a*q2_a
    for window_size in range(w_size):
        tmp = 1 - pow(pr,window_size)
        if not tmp == 0:
            tmp = math.exp(m * n * math.log(tmp))
        ans.append(tmp)
    plt.plot(ans, label = lab)
    return ans

def GetProb(content):
    prob = {}
    for i in content:
        prob[i] = prob.get(i,0) + 1
    total = sum(prob.values())
    for i in prob.keys():
        prob[i] = float(prob.get(i)/float(total))
    return prob

def GetSequence(content):
    sequence_dict = {}
    for i in range(len(content)-8):
        window = content[i:i+8]
        pred = content[i+8]
        current_counts_dict = sequence_dict.get(window,{})
        current_counts_dict[pred] = current_counts_dict.get(pred,0) + 1
        sequence_dict[window] = current_counts_dict
    return sequence_dict

def GetProbDist(sequence_dict):
    prob_dict = {}
    for i in sequence_dict.keys():
        cur_sum = sum(sequence_dict[i].values())
        cur_prob_dict = {}
        for j in sequence_dict[i].keys():
            cur_prob_dict[j] = sequence_dict[i][j]/cur_sum
        prob_dict[i] = cur_prob_dict
    return prob_dict

def GenerateNewSeq(dict):
    temp_seq = ['A','C','G','T']
    # if new sub sequence not present in original sequence
    if len(dict) == 0:
        global ze
        ze = ze + 1
        return temp_seq[random.randint(0,3)]
    r = random.uniform(0,1)
    for i in sorted(dict.items(), key = lambda kv:(kv[1], kv[0]), reverse = True):
        if r <= i[1]:
            return i[0]
        r = r - i[1]

def GenerateSeq(prob_dict, content, seqSize):
    cur_state = content[:8]
    # cur_state = GetIndependentSeq(8)
    new_sequence = cur_state
    while len(new_sequence) < seqSize:
        gen = GenerateNewSeq(prob_dict.get(cur_state,{}))
        new_sequence += gen
        cur_state = cur_state[1:] + gen
    return new_sequence

def CreateDict(gnome, window_size):
    new_dict = {}
    for i in range(len(gnome) - window_size):
        window = gnome[i:(i+window_size)]
        new_dict[window] = new_dict.get(window,0) + 1
    return new_dict

def GetDict(new_sequence, w_size, default_window_slices, index):
    list_window_slices = {}
    for window_size in range(w_size):
        list_window = {}
        list_window['b'] = default_window_slices.get(window_size).get('b')
        list_window[index] = CreateDict(new_sequence,window_size)
        list_window_slices[window_size] = list_window
    return list_window_slices

def GetDefaultDict(w_size, content):
    list_window_slices = {}
    for window_size in range(w_size):
        list_window = {}
        list_window['b'] = CreateDict(content,window_size)
        list_window_slices[window_size] = list_window
    return list_window_slices

def GetIntersection(list_window_slices):
    all_ratio = {}
    for window_size in list_window_slices.keys():
        window_slice = list_window_slices.get(window_size)
        base = window_slice.get('b').keys()
        del window_slice['b']
        for vir_k,vir in window_slice.items():
            intersections = sum([1 for k in vir if k in base])
            temp = all_ratio.get(vir_k,[])
            temp.append(1 - float(intersections)/iid_samples)
            all_ratio[vir_k] = temp
    return all_ratio

def GetIndependentSeq(n_size):
    temp_seq = ['A','C','G','T']
    newContent = ''
    for i in range(n_size):
        newContent += temp_seq[random.randint(0,3)]
    return newContent

def GetIIDDict(w_size, default_window_slices):
    list_window_slices = {}
    index = 0
    for window_size in range(w_size):
        list_window = {}
        list_window['b'] = default_window_slices.get(window_size).get('b')
        list_window[index] = [GetIndependentSeq(window_size) for i in range(iid_samples)]
        list_window_slices[window_size] = list_window
    return list_window_slices

def GetDictSW(new_sequence, window_size, default_window_slices, index, list_window_slices):
    list_window = {}
    list_window['b'] = default_window_slices.get(window_size).get('b')
    list_window[index] = CreateDict(new_sequence,window_size)
    list_window_slices[window_size] = list_window
    return list_window_slices

def GetIntersectionSW(list_window_slices, window_size):
    window_slice = list_window_slices.get(window_size)
    base = window_slice.get('b').keys()
    del window_slice['b']
    mutation = window_slice.get(0).keys()
    inter = set(base).intersection(set(mutation))
    return inter

def GetIntersectionB(new_sequence, default_window_slices, list_window_slices, max_window_size):
    window_size = 0
    low = 0
    max = max_window_size
    while True:
        window_size = int((low+max)/2)
        if max <= low:
            break
        list_window_slices = GetDictSW(new_sequence, window_size, default_window_slices, 0, list_window_slices)
        inter = GetIntersectionSW(list_window_slices, window_size)
        if len(inter) == 0:
            max = window_size - 1
        else:
            low = window_size + 1
    list_window_slices = GetDictSW(new_sequence, window_size, default_window_slices, 0, list_window_slices)
    inter = GetIntersectionSW(list_window_slices, window_size)
    if len(inter) > 0:
        window_size += 1
    return window_size

def GetIntersectionL(new_sequence, default_window_slices, list_window_slices, max_window_size):
    max_match = max_window_size
    for window_size in range(max_window_size-1,0,-1):
        list_window_slices = GetDictSW(new_sequence, window_size, default_window_slices, 0, list_window_slices)
        inter = GetIntersectionSW(list_window_slices, window_size)
        if len(inter) > 0:
            break
        max_match = window_size
    return max_match

def GetSimRes(content, default_window_slices, num_samples, max_window_size,lab):
    seq = GetSequence(content)
    prob_dist = GetProbDist(seq)
    sim_res = {}
    for i in range(max_window_size):
        sim_res[i] = 0

    for index in range(num_samples):
        # for markov model
        new_sequence = GenerateSeq(prob_dist, content, len(content))
        # for sliding window IID model
        # new_sequence = GetIndependentSeq(len(content))

        # list_window_slices = GetDict(new_sequence, max_window_size, default_window_slices, 0)
        list_window_slices = {}

        max_match = GetIntersectionB(new_sequence, default_window_slices, list_window_slices, max_window_size)

        print('sample:',index,', max windows size match:',max_match)

        for i in range(max_match):
            sim_res[i] += 1

        # match = SequenceMatcher(None, new_sequence, content).find_longest_match(0, len(new_sequence), 0, len(content))
        # print(match)
    for i in range(max_window_size):
        sim_res[i] = 1 - float(sim_res[i])/num_samples
    sim_res = list(sim_res.values())
    # plt.plot(sim_res, label = lab)
    return sim_res

def CreateSequence(content, filename, firstLine = "Generated sequence with Markov Model \n"):
    seq = GetSequence(content)
    prob_dist = GetProbDist(seq)
    new_sequence = GenerateSeq(prob_dist, content, len(content))
    default_window_slices = GetDefaultDict(max_window_size, new_sequence)
    f = open(filename, "w")
    f.write(firstLine)
    f.write(new_sequence)
    f.close()
    return default_window_slices

# get the contents from file
def getContents(filename):
    contents = ''
    with open(filename) as f:
        fileLines = f.readlines()
        # remove first line
        for  x in fileLines[1:]:
            contents += x.strip()
    return contents

def getPoint(point, arr):
    xub,xlb,yub,ylb = 0, 0, 0, 0
    for idx,i in enumerate(arr):
        if i < point:
            xlb = idx
            ylb = i
        if i > point:
            yub = i
            xub = idx
            break
    m = float(yub-ylb)/float(xub-xlb)
    x = xlb + float(point - ylb)/m
    return x

def main(model):
    global hostfilename, targetfilename, max_window_size, cp, points, num_samples
    content_host = getContents(hostfilename)
    gc_host = GetProb(content_host)
    default_window_slices = GetDefaultDict(max_window_size, content_host)
    # iid model
    if model == 'i':
        list_window_slices = GetIIDDict(max_window_size,default_window_slices)
        val = GetVal(1,len(content_host),GetProb('AGCT'),gc_host, max_window_size, 'IIDformula')
        all_ratio = GetIntersection(list_window_slices)
        print(all_ratio[0])
        print(val)
        plt.plot(all_ratio[0], label = 'IID')
        plt.xlabel('window size')
        plt.ylabel('probability of no intersections')
        plt.legend()
        plt.savefig('intersectionsIID.png')
        plt.show()

    # sliding window model
    else:
        # compare with self
        if model == 's':
            content_target = content_host
            gc_target = gc_host
        # compare with another sequence
        elif model == 'n':
            content_target = getContents(targetfilename)
            gc_target = GetProb(content_target)
        sim_res = GetSimRes(content_target, default_window_slices, num_samples, max_window_size, 'Sliding:Simulation')

        val = GetVal(len(content_host), len(content_target), gc_host, gc_target, max_window_size, 'Sliding:Formula')

        print('Values from Simulation: ', sim_res)
        print('Values from formula: ', val)
        print(ze)
        print('For theoretical values -')
        for p in points:
            print(p, ':', getPoint(p, val))
        print('For simulation values -')
        for p in points:
            print(p, ':', getPoint(p, sim_res))

        plt.xlabel('window size')
        plt.ylabel('probability of no intersections')
        plt.legend()
        plt.savefig('intersectionsSliding.png')
        plt.show()

if __name__ == "__main__":
    # cp.enable()
    main(sys.argv[len(sys.argv)-1])
    # cp.disable()
    # cp.print_stats()
