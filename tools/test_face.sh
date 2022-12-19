# test shell for all face data
root='$HOME/face_det/'

# test nonghang
echo "test nonghang km..."
python eval/run.py cfgs/face_solver.py facedet_badcase_sideface/image_kunming_bank_imgs_test.lst result/result.txt
python eval/mAP_PR.py ${root} facedet_badcase_sideface/image_kunming_bank_imgs_test.lst None result/result.txt

# # test sense
# echo "test sense..."
# python eval/run.py cfgs/face_solver.py face_monitor/test.lst result/result.txt
# python eval/mAP_PR.py ${root} face_monitor/test.lst None result/result.txt

# # test persion
# echo 'test persion face...'
# python eval/run.py cfgs/face_solver.py person_face/personfacetest.lst result/result.txt
# python eval/mAP_PR.py ${root} person_face/personfacetest.lst None result/result.txt

# # test license
# echo 'test license face...'
# python eval/run.py cfgs/face_solver.py face_license/test2.lst result/result.txt
# python eval/mAP_PR.py ${root} face_license/test2.lst None result/result.txt

# # test carface
# echo "test carface..."
# python eval/run.py cfgs/face_solver.py car_face_test/car_face_test.lst result/result.txt
# python eval/mAP_PR.py ${root} car_face_test/car_face_test.lst None result/result.txt

# # test maskface
# echo "test xjmask..."
# python eval/run.py cfgs/face_solver.py face_det_mask/face_det_mask200128_test.lst result/result.txt
# python eval/mAP_PR.py ${root} face_det_mask/face_det_mask200128_test.lst None result/result.txt
