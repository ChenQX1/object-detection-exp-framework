import torch


def box_center(boxes):
    xc = (boxes[:,0]+boxes[:,2])/2.
    yc = (boxes[:,1]+boxes[:,3])/2.
    xc = xc.view(-1,1)
    yc = yc.view(-1,1)
    center = torch.cat([xc,yc],dim=1)
    return center

'''
box1:row,box2:col
'''
def bboxs_distance(boxs1,boxs2):
    c1 = box_center(boxs1)
    #print(c1)
    c2 = box_center(boxs2)
    distance = (c1[:, None, :] - c2[None, :, :]).pow(2).sum(-1).sqrt()
    return distance


def main():
    box1 = torch.tensor([[1,1,2,2],[3,3,4,4]])
    box2 = torch.tensor([[1,1,2,2],[3,3,4,4]])
    mat = bboxs_distance(box1,box2)
    print(mat)
    idx = torch.argmin(mat,dim=1)
    print(idx)

if __name__ == "__main__":
    main()
    

