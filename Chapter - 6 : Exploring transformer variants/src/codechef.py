# array = [3,4,4,3,3,4]

# value = None
# for i in range(len(array)-1):
#     if value==None or value < array[i]:
#      value = max(array[i],array[i+1])
#     if (array[i]+1) == value:
#         array[i] +=1
#     elif (array[i+1]+1)==value:
#         array[i+1] +=1

# print(array)



cities = 5
one_sided_roads= 5
connectivity = [(1,2),(3,2),(4,3),(5,4),(5,3)]

coverage = {}
for wareh_c in range(1,cities+1):
    #possible
    #coverage, how many i may miss
    flag = False
    coverage[wareh_c] = 0
    for x,y in connectivity:
        if x==wareh_c:
            coverage[wareh_c] +=1