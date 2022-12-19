import os

root_dir = '/face_det'
#root_dir = '~/ljw/dgface_det/face_det'

train_list = [
    # org trianset
    'sndgcar_train.lst',
    'persontrain.txt',
    # dms data
    'DMS_0711/train.lst',
    'dms_20181225/train.lst',
    # mafa data
    'face_det_hardcase/face_det_hardcase_mafa.lst',
    # face with mask            
    'face_det_mask/face_det_mask200128_train.lst',
    'face_det_mask/face_det_mask200131_std.lst',
    'face_det_mask/face_det_mask200203_std.lst',
    # badcase sence
    'facedet_badcase_nanjing/facedet_badcase_nanjing.lst',         #nanjing
    'facedet_badcase_sideface/facedet_badcase_sideface0714.lst',   #nonghang
    'facedet_badcase_sideface/sty_1kscene_with5k_sub3w_1w0723.lst', #shentongyun
    # badcase tiny
    'facedet_animal/pair.lst',
    'facedet_wheel/pair.lst',
    # psuduo label 
    'facedet_profile/pair.lst',   #(with resnet50_dcn)
    'facedet_lowhead2/pair.lst',  #(with resnet50_dcn)
    #'facedet_nonghang/pair.lst',  #(with resnet50_dcn)
    #'facedet_shinei/pair.lst',   #(with resnet50_dcn)
    # psuduo label (zhaji data) 
    #'model_label_0916/pair.lst',
    #'model_label_10w/pair.lst',
]

tagnames = {
    'face':0,
}
