import random
import matplotlib.pyplot as plt
import math
import sys
# import pickle
# from difflib import SequenceMatcher
# import cProfile
#
# cp = cProfile.Profile()


filename = 'sequence.fasta'
filename1 = 'sequence_f.fasta'
filename2 = 'sequence_c.fasta'
max_window_size = 31
num_samples = 1000
iid_samples = 1000
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
    global max_window_size
    window_slice = list_window_slices.get(window_size)
    base = window_slice.get('b').keys()
    del window_slice['b']
    mutation = window_slice.get(0).keys()
    inter = set(base).intersection(set(mutation))
    return inter

# get the contents from file
def getContents(filename):
    contents = ''
    with open(filename) as f:
        fileLines = f.readlines()
        # remove first line
        for  x in fileLines[1:]:
            contents += x.strip()
    return contents

def main(model):
    global filename, filename1, max_window_size, num_samples, cp
    content_bacteria = getContents(filename)
    content_bacteria_circular = getContents(filename2)
    # content_fungi = getContents(filename1)
    gc_bacteria = GetProb(content_bacteria)
    gc_bacteria_circular = GetProb(content_bacteria_circular)
    # gc_fungi = GetProb(content_fungi)
    default_window_slices = GetDefaultDict(max_window_size, content_bacteria)

    # iid model
    if model == 'i':
        list_window_slices = GetIIDDict(max_window_size,default_window_slices)
        val = GetVal(1,len(content_bacteria),GetProb('AGCT'),gc_bacteria, max_window_size, 'IIDformula')
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
    elif model == 's':
        seq = GetSequence(content_bacteria)
        prob_dist = GetProbDist(seq)
        sim_res = {}
        for i in range(max_window_size):
            sim_res[i] = 0

        for index in range(num_samples):
            # index variable is used to store the results for different sims. Due to mem constraints, it is not currently. [index = 0 is used]

            # for markov model
            new_sequence = GenerateSeq(prob_dist, content_bacteria, len(content_bacteria))
            # for sliding window IID model
            # new_sequence = GetIndependentSeq(len(content_bacteria))

            # list_window_slices = GetDict(new_sequence, max_window_size, default_window_slices, 0)
            list_window_slices = {}

            for window_size in range(max_window_size-1,0,-1):
                list_window_slices = GetDictSW(new_sequence, window_size, default_window_slices, 0, list_window_slices)
                inter = GetIntersectionSW(list_window_slices, window_size)
                if len(inter) > 1:
                    break
                max_match = window_size

            print('sample:',index,', max windows size match:',max_match)

            for i in range(max_match):
                sim_res[i] += 1

            # match = SequenceMatcher(None, new_sequence, content_bacteria).find_longest_match(0, len(new_sequence), 0, len(content_bacteria))
            # print(match)
        for i in range(max_window_size):
            sim_res[i] = 1 - float(sim_res[i])/num_samples
        val = GetVal(len(content_bacteria), len(content_bacteria), gc_bacteria, gc_bacteria, max_window_size, 'Slidingformula')
        print(list(sim_res.values()))
        print(val)
        print(ze)
        plt.plot(list(sim_res.values()), label = 'Sliding:Simulation')
        plt.xlabel('window size')
        plt.ylabel('probability of no intersections')
        plt.legend()
        plt.savefig('intersectionsSliding.png')
        plt.show()
    # compare sequences from 2 different bacterias
    elif model == 'n':
        seq_b = GetSequence(content_bacteria)
        seq_c = GetSequence(content_bacteria_circular)
        prob_dist_b = GetProbDist(seq_b)
        prob_dist_c = GetProbDist(seq_c)
        sim_res = {}
        for i in range(max_window_size):
            sim_res[i] = 0
        # new_sequence_b = GenerateSeq(prob_dist_b, content_bacteria, len(content_bacteria))
        # default_window_slices = GetDefaultDict(max_window_size, new_sequence_b)
        for index in range(num_samples):
            new_sequence_c = GenerateSeq(prob_dist_c, content_bacteria_circular, len(content_bacteria_circular))
            # new_sequence_c = GenerateSeq(prob_dist_b, content_bacteria, len(content_bacteria))
            list_window_slices = {}

            for window_size in range(max_window_size-1,0,-1):
                list_window_slices = GetDictSW(new_sequence_c, window_size, default_window_slices, 0, list_window_slices)
                inter = GetIntersectionSW(list_window_slices, window_size)
                if len(inter) > 1:
                    break
                max_match = window_size

            print('sample:',index,', max windows size match:',max_match)
            for i in range(max_match):
                sim_res[i] += 1

            # match = SequenceMatcher(None, new_sequence, content_bacteria).find_longest_match(0, len(new_sequence), 0, len(content_bacteria))
            # print(match)
        for i in range(max_window_size):
            sim_res[i] = 1 - float(sim_res[i])/num_samples
        val = GetVal(len(content_bacteria), len(content_bacteria), gc_bacteria, gc_bacteria, max_window_size, 'Slidingformula')
        print(list(sim_res.values()))
        print(val)
        print(ze)
        plt.plot(list(sim_res.values()), label = 'Sliding:Simulation')
        plt.xlabel('window size')
        plt.ylabel('probability of no intersections')
        plt.legend()
        plt.savefig('intersectionsSliding.png')
        plt.show()
    # cp.disable()
    # cp.print_stats()

# to run, python dr.py [i|s] : i = iid model,s = sliding window model, python dr.py for results of sliding window simulations.
if __name__ == "__main__":
    main(sys.argv[len(sys.argv)-1])
