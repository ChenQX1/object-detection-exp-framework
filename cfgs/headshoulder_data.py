import os
root_dir = '/train/trainset/1/mt2/Gucci_head_shoulder/train'
train_list = [
    'txt_info/trainval_2.5w_dgdet.txt',
]

test_root_dir = '/train/trainset/1/mt2/Gucci_head_shoulder/val'
#test_list = [
#    'txt_info/test1.txt',
#    'txt_info/test2.txt',
#]

test_list = ['txt_info/CD_side_1125_dgdet.txt',]

tagnames = {
    'head_shoulder':0,
}

# # for calculate P and R in evaluation phase
vis_flag = False 
vis_path = './pred_headshoulder'    # './pred_headshoulder/test1/    ./pred_headshoulder/test2/'
js_dir = './result_0301'  # './result_test1.json   ./eval_test1.json    ./result_test2.json    ./eval_test2.json

'''
# # for run detection demo for single image or dir files (without GT)
img_path = 'raw.jpg'    # img_path or dir filename   eg: img_path = vis_path+'/'+'demo/input'
# img_path = '/train/trainset/1/mt2/Gucci_head_shoulder/val/CD_side_1125_images'
result_path = vis_path+'/'+'demo/output'
'''



