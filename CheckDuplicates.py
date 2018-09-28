# import sys
#
# f = open(sys.argv[1], "r")
#
# insts = []
# for l in f:
#     insts.append(l)
# print insts

N = 1044
insts = []
for i in range(0, N):
    line = raw_input()
    insts.append(line)
# print insts

for i in range(0, N):
    for j in range(i + 1, N):
        if (insts[i] is insts[j]):
            print "vixi"
