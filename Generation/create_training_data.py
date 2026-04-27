import os, random

number_files = 200

random_numbers = tuple([random.randint(1, 20)/10 for i in range(number_files)])


for i in range(number_files):
    os.system(f'C:/Users/Maros/AppData/Local/Microsoft/WindowsApps/python3.11.exe save_created_ask.py --outfile Training_data4/{"{:06d}".format(i)}.complex --bits-outfile Training_data4/{"{:06d}".format(i)}_text.txt --noise {random_numbers[i]} --freq {random.randint(10000, 500000)} --numbits {random.randint(1, 32)} --bits-per-sample {random.randint(32, 256)}')
    if i % (number_files // 100) == 0:
        print(int((i / number_files)*100))

