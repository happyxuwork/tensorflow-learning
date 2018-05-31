

import os
# str = 'fasdfasdf'
# lis = ['yes/jj/xu','mm/kk/qiang']
# lis = lis + [str]
# for i in lis:
#     # print(os.path.basename(i))
#     print(i)

# if not os.path.exists("F:/alltoNPZ1"):
#     os.makedirs("F:/alltoNPZ1")
# str = "xx"
# file_name = os.path.join("sss",str+".npz")
# print(file_name)
image_all_to_npz_path = "F:/alltoNPZ/"
npz_sub_dirs = [x[2] for x in os.walk(image_all_to_npz_path)]
# print(npz_sub_dirs)
# print(npz_sub_dirs[0])
# npz_sub_dirs[0].remove('realimg.npz')
# print(npz_sub_dirs[0])

for sub_dir, i in zip(npz_sub_dirs[0], range(4)):
    print(sub_dir+"        "+str(i))
