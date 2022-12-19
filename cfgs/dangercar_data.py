import os

root_dir = '/danger_car/data'

train_list = [
    'train1/pair.lst',
    'train2/pair.lst',
    'train_2020/pair.lst',
]


test_list = [
    'test/pair.lst',
]

tagnames = {
    'risk_mark_triangle':0,
    'dangerous_signs':1,
}

js_dir = 'result'