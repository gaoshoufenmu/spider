
def removeDuplicates(height) -> int:
    if not height:
        return 0
    if len(height)<3:
        return 0
    rain=0
    indices=[]
    for i in range(len(height)):
        if i==0 and height[i]>height[i+1]:
            indices.append(i)
        elif i==len(height)-1 and height[i]>height[i-1]:
            indices.append(i)
        elif height[i]>=height[i-1] and height[i]>=height[i+1]:
            indices.append(i)
        while len(indices)>=3:
            if height[indices[-1]]>=height[indices[-2]] and \
                height[indices[-2]]<=height[indices[-3]]:
                indices[-2]=indices[-1]
                indices.pop()
            else:
                break
    
    for i in range(len(indices)-1):
        left=indices[i]
        right=indices[i+1]
        minimum=min(height[left], height[right])
        for j in range(left, right+1):
            bucket=minimum-height[j] if minimum>height[j] else 0
            rain+=bucket
    return rain

if __name__=='__main__':
    l=[8,8,1,5,6,2,5,3,3,9]
    c=removeDuplicates(l)
    print(c)