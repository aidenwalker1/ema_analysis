from datetime import datetime
import csv
import matplotlib.pyplot as plt
import numpy as np

def read_ihs(path, id) :
    id *= 1000
    data = []
    last_id = None 
    first_time = None

    file = open(path, 'r')
    file.readline()
    reader = csv.reader(file)
    cur_id = 0
    day_dict = {}
    total_days = 0
    for line in reader :
        r1, r2, r3 = int(line[2]), 6 - int(line[6]), int(line[7])
        if last_id == None :
            last_id = line[0]
            first_time = datetime.strptime(line[1], "%Y-%m-%d %H:%M:%S")
        
        if last_id != line[0] :
            cur_id += 1
            first_time = datetime.strptime(line[1], "%Y-%m-%d %H:%M:%S")
            last_id = line[0]
            day_dict = {}
            total_days = 0
        cur_time = datetime.strptime(line[1], "%Y-%m-%d %H:%M:%S")
        cur_min = (cur_time.hour*60) + cur_time.minute
        
        cur_hour = cur_time.hour
        if day_dict.get(cur_time.day) == None :
            day_dict[cur_time.day] = total_days
            total_days += 1
        cur_day = day_dict[cur_time.day] 
        delta = int((cur_time - first_time).total_seconds() / 60)
        data.append([id + cur_id, delta, cur_min,cur_hour,cur_day,r1, r2, r3])
    file.close()
    return data

def read_data(path, id, delims,date_format,functions,wacky=False) :
    id *= 1000
    file = open(path, 'r')
    stop = False
    cur_id = 0
   
    data = []

    first_time = None
    last_id = None
    line = None

    day_dict = {}
    total_days = 0

    while file:
        r1 = -1
        r2 = -1
        r3 = -1
        prompt_id = None
        
        different_id = False

        while r1 == -1 or r2 == -1 or r3 == -1 :
            line = file.readline()

            if line == '' :
                stop = True
                break

            cur_prompt = line[:line.index(',')]
            if cur_prompt == '11900231' :
                print('stop')

            if prompt_id == None :
                prompt_id = cur_prompt
            
            if cur_prompt != prompt_id :
                different_id = True
                break

            if delims[0] in line:
                try :
                    r1 = int(line[line.index('(')+1])
                    r1 = functions[0](r1)
                except: 
                    r1 = -2
            elif delims[1] in line :
                try :
                    r2 = int(line[line.index('(')+1])
                    r2 = functions[1](r2)
                except: 
                    r2 = -2
            elif delims[2] in line :
                try :
                    r3 = int(line[line.index('(')+1])
                    r3 = functions[2](r3)
                except:
                    r3 = -2
        if stop :
            break
        if different_id :
            #if r1 == -2 or r2 == -2 or r3 == -2 :
            print(line)
            continue
        
        if r1 == -2 or r2 == -2 or r3 == -2 :
           if not wacky :
                continue

        line = line.split(',')

        if '.' in line[6] :
            line[6] = line[6][:line[6].index('.')]
        if '.' in line[7] :
            line[7] = line[7][:line[7].index('.')]
        if '.' in line[4] :
            line[4] = line[4][:line[4].index('.')]
        if '.' in line[5] :
            line[5] = line[5][:line[5].index('.')]
        
        if last_id == None :
            last_id = line[2]
            first_time = datetime.strptime(line[6] + ' ' + line[7], date_format)
        
        if last_id != line[2] :
            cur_id += 1
            first_time = datetime.strptime(line[6] + ' ' + line[7], date_format)
            last_id = line[2]
            day_dict = {}
            total_days = 0
        start_time = datetime.strptime(line[4] + ' ' + line[5], date_format)
        cur_time = datetime.strptime(line[6] + ' ' + line[7], date_format)
        cur_min = (cur_time.hour*60) + cur_time.minute
        
        cur_hour = cur_time.hour
        if day_dict.get(cur_time.day) == None :
            day_dict[cur_time.day] = total_days
            total_days += 1
        cur_day = day_dict[cur_time.day] 
        delta = int((cur_time - first_time).total_seconds() / 60)
        response_time = (cur_time - start_time).total_seconds()

        data.append([id + cur_id, delta, cur_min,cur_hour,cur_day,int(response_time),r1, r2, r3])
    file.close()
    return data

def read_prompts(path, id, delims,date_format) :
    id *= 1000
    file = open(path, 'r')
    file.readline()
    stop = False
    cur_id = 0
   
    data = []

    first_time = None
    last_id = None

    day_dict = {}
    total_days = 0

    
    line = file.readline()
    cur_data = []

    while line != '':
        prompt = line[:line.index(',')]
        old_prompt = prompt
        cur_data = []

        while prompt == old_prompt and line != '' :
            old_prompt = prompt
            cur_data.append(line)
            line = file.readline()

            if line != '' :
                prompt = line[:line.index(',')]
        
        if line == '' :
            break
        
        r1 = None
        r2 = None
        r3 = None

        for l in cur_data :
            row = l.split(',')
            if delims[0] in row[-2]:
                r1 = 0 if row[-1] == '' else 1
            elif delims[1] in row[-2] :
                r2 = 0 if row[-1] == '' else 1
            elif delims[2] in row[-2] :
                r3 = 0 if row[-1] == '' else 1

        if r1 == None or r2 == None or r3 == None :
            print(line)
            continue
        next_line = line
        line = cur_data[-1]
        line = line.split(',')

        if '.' in line[6] :
            line[6] = line[6][:line[6].index('.')]
        if '.' in line[7] :
            line[7] = line[7][:line[7].index('.')]
        if '.' in line[4] :
            line[4] = line[4][:line[4].index('.')]
        if '.' in line[5] :
            line[5] = line[5][:line[5].index('.')]
        
        if last_id == None :
            last_id = line[2]
            first_time = datetime.strptime(line[6] + ' ' + line[7], date_format)
        
        if last_id != line[2] :
            cur_id += 1
            first_time = datetime.strptime(line[6] + ' ' + line[7], date_format)
            last_id = line[2]
            day_dict = {}
            total_days = 0
        start_time = datetime.strptime(line[4] + ' ' + line[5], date_format)
        cur_time = datetime.strptime(line[6] + ' ' + line[7], date_format)
        cur_min = (cur_time.hour*60) + cur_time.minute
        
        cur_hour = cur_time.hour
        if day_dict.get(cur_time.day) == None :
            day_dict[cur_time.day] = total_days
            total_days += 1
        cur_day = day_dict[cur_time.day] 
        delta = int((cur_time - first_time).total_seconds() / 60)
        response_time = (cur_time - start_time).total_seconds()

        data.append([id + cur_id, delta, cur_min,cur_hour,cur_day,int(response_time),r1, r2, r3])

        line = next_line
    file.close()
    return data


linetot = 0
errortot = 0


import read_ema_2
def verify_emas(path, delims) :
    global linetot
    global errortot
    file = open(path, 'r')
    file.readline()
    reader = csv.reader(file)
    errors = 0
    errors2 = 0
    line = 0
    line2 = 0
    d = {}
    d2 = {}
    d3 = {}
    qs = {}
    times = [0] * 24#480
    times2 = [0] * 280
    times3 = [0] * 480

    day_dict = {}
    total_days = 0

    for row in reader :
        st = ''.join(row)

        if d2.get(row[0]) == None :
            d2[row[0]] = 0
        
            
        d2[row[0]] += 1
        if '.' in row[6] :
            row[6] = row[6][:row[6].index('.')]
        if '.' in row[7] :
            row[7] = row[7][:row[7].index('.')]

        time = row[6] + ' ' + row[7]
        time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')

        if d3.get(row[2]) == None :
            d3[row[2]] = 1
            day_dict = {}
            total_days = 0

        if day_dict.get(time.day) == None :
            day_dict[time.day] = total_days
            total_days += 1

        if qs.get(row[-2]) == None :
            qs[row[-2]] = 0
        qs[row[-2]] +=1
        if row[-1] == '' :
            if d.get(row[0]) == None :
                
                d[row[0]] = 0
            d[row[0]] += 1

            day = day_dict[time.day] 

            t = time.hour #+ (day*24)
            t2 = t + (day*14)

            times[t] += 1
            if t >=8 and t < 22 :
                times2[t2-8] += 1
            times3[t + (day*24)] += 1
            
            if delims[0] in st or delims[1] in st or delims[2] in st :
                errors += 1
                errortot += 1
            errors2 += 1
            #print(row)
        if delims[0] in st or delims[1] in st or delims[2] in st :
            line += 1
        line2 += 1
        linetot += 1
    
    i = len(times2)-1
    while times2[i] == 0:
        del times2[i]
        i-=1

    plt.bar(np.arange(0,len(times2)), times2)
    plt.show()

    plt.bar(np.arange(0,len(times)), times)
   
    plt.title("Response Failure Times over Study")
    plt.xlabel("Time (Days)")
    plt.xticks([0, 3, 6, 9, 12, 15, 18, 21, 24], ['12 am', '3 am', '6 am', '9 am', '12 pm', '3 pm', '6 pm', '9 pm','12 am'])
    #ticks = [i for i in range(len(times3)) if i % 24 == 0]
    #plt.xticks(ticks, np.array(ticks) // 24)
    plt.ylabel("Prompt Failures")
    plt.show()
    
    times = times[8:21]#[t for t in times if t > 0]
    cur = read_ema_2.make_features(times)
    read_ema_2.print_data(cur, "Times")

    cur = read_ema_2.make_features(times2)
    read_ema_2.print_data(cur, "Study")


    l = [0] * 20
    l2 = [0] * 20
    print(qs)
    for k,v in qs.items() :
        print(f'{k} {v}')
    
    for key, value in d2.items() :
        
        if d.get(key) != None :
            if value == 1 :
                x = 1
            l[value-1] += 1 
    for key, value in d.items() :
        l2[value-1] += 1 

    # plt.bar(np.arange(1,len(l)+1), l)
   
    # plt.title("Prompt Length vs Amount of Failures (GSUR)")
    # plt.xlabel("Prompt Length")
    # plt.ylabel("Prompt Failures")
    # plt.xticks([1,2,3,4,5,6,7,8,9])
    # plt.show()
    print(l)
    print(l2)
    print(f"Path:{path}, errors: {errors} {errors2}, lines: {line} {line2}, ratio: {errors/line}, ratio2: {errors2/line2}")

    file.close()


folder = "./ema_nback/"
verify_emas(folder + "rwcs_ema.csv", ["sharp", "fatigue", "stress"])
verify_emas(folder + "gsur_2.0_ema.csv", ["sharp", "fatigue", "stress"])


six = lambda x : 6 - x
same = lambda x : x
rnd = lambda x : int(round(x*(5.0/7.0)))
six_rnd = lambda x : 6 - rnd(x)
six_add = lambda x : 6 - add(x)
add = lambda x : x + 1

default_format = "%Y-%m-%d %H:%M:%S"
dod_format = "%m/%d/%Y %H:%M:%S"

rwcs = read_prompts(folder + "rwcs_ema.csv", 8, ["sharp", "fatigue", "stress"], default_format)
gsur_2 = read_prompts(folder + "gsur_2.0_ema.csv", 6, ["sharp", "fatigue", "stress"], default_format)

chile = read_data(folder + "chile_ema.csv", 1, ["sharp", "fatigued", "stressed"], default_format,[rnd,six_rnd,six_rnd]) 
dod = read_data(folder + "dod_ema.csv", 2, ["sharp", "independent", "over"],dod_format,[rnd,rnd,six_rnd])   
dyad = read_data(folder + "dyad_ema.csv", 3, ["confident", "going", "anxious"], default_format,[add,add,six_add])    
fau = read_data(folder + "fau_ema.csv", 4, ["sharp", "fatigue", "anxious"],default_format,[same,six,six])   
func =  read_data(folder + "func_ema.csv", 5, ["motivated", "fatigue", "stress"],default_format,[same,six,six])   
gsur = read_data(folder + "gsur_ema.csv", 6, ["sharp", "fatigue", "stress"],default_format,[same,six,six])   
ucd = read_data(folder + "ucd_ema.csv", 7, ["sharp", "fatigue", "stress"], default_format,[same,six,six])   

ihs = read_ihs(folder + "ihs_ema.csv", 9)

all_data = chile + dod + dyad + fau + func + gsur + ucd #+ ihs

file = open('ema_new.csv', 'w',newline="")

writer = csv.writer(file)
writer.writerow(["id", "time_elapsed", "minute", "hour", "day","response_time","sharp","fatigue","stress"])
writer.writerows(all_data)
file.close()

file = open('rwcs_new.csv', 'w',newline="")

writer = csv.writer(file)
writer.writerow(["id", "time_elapsed", "minute", "hour", "day","response_time","sharp","fatigue","stress"])
writer.writerows(rwcs)
file.close()

file = open('gsur_new.csv', 'w',newline="")

writer = csv.writer(file)
writer.writerow(["id", "time_elapsed", "minute", "hour", "day","response_time","sharp","fatigue","stress"])
writer.writerows(rwcs)
file.close()