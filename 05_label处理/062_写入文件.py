import os

path  = r'C:\01_data\text\\'
files = os.listdir(path)


for file in files:
    text = open(path + file)
    with open(file,'w+') as wf:
        wf.write('389 518 450 548 0')


# def w_file(filepath):
#     with open(filepath, 'w') as wf:
#         wf.write('389 518 450 548 0')
#         # wf.write('do you want to take a trip.')
#
#
# w_file('wfile.txt')