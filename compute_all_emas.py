from datetime import datetime
import csv
import matplotlib.pyplot as plt
import numpy as np
from read_and_graph_data import make_features, print_data
import re

linetot = 0
errortot = 0

def read_ihs(path:str, id:int) -> list:
    """
        Reads the ihs file
    
        Parameters
        ----------
        path : str
            Path to csv file
        id : int
            Id to save in list
    
        Returns
        -------
        list
            All ihs data
    """

    # allows for 1000 ids per study
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

    # go through each line
    for line in reader :
        # get responses
        r1, r2, r3 = int(line[2]), 6 - int(line[6]), int(line[7])

        # get first id and time
        if last_id == None :
            last_id = line[0]
            first_time = datetime.strptime(line[1], "%Y-%m-%d %H:%M:%S")
        
        # check if new person
        if last_id != line[0] :
            # go to next id, reset first time and day
            cur_id += 1
            first_time = datetime.strptime(line[1], "%Y-%m-%d %H:%M:%S")
            last_id = line[0]
            day_dict = {}
            total_days = 0
        
        # get time, minute, hour
        cur_time = datetime.strptime(line[1], "%Y-%m-%d %H:%M:%S")
        cur_min = (cur_time.hour*60) + cur_time.minute
        
        cur_hour = cur_time.hour

        # get current day (0=first day)
        if day_dict.get(cur_time.day) == None :
            day_dict[cur_time.day] = total_days
            total_days += 1
        
        cur_day = day_dict[cur_time.day] 

        # get time since study started
        delta = int((cur_time - first_time).total_seconds() / 60)

        data.append([id + cur_id, delta, cur_min,cur_hour,cur_day,r1, r2, r3])
    file.close()
    return data

def read_data(path, id, delims,date_format,functions) :
    """
        Reads the regular file
    
        Parameters
        ----------
        path : str
            Path to csv file
        id : int
            Id to save in list
        delims: list
            list of strings to denote wanted categories
        date_format: str
            strptime format
        functions: list
            list of functions to apply to each response
    
        Returns
        -------
        list
            All ema data
    """

    id *= 1000
    file = open(path, 'r')
    file.readline()
    cur_id = 0
   
    data = []

    first_time = None
    last_id = None

    day_dict = {}
    total_days = 0

    
    line = file.readline()
    cur_data = []

    # goes through each line
    while line != '':
        prompt = line[:line.index(',')]
        old_prompt = prompt
        cur_data = []

        # gets current prompt values
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

        # extracts responses from prompt
        for l in cur_data :
            row = l.split(',')

            if row[-1] == '\n' :
                continue

            # checks if has correct word, then checks if not empty
            if delims[0] in row[-2] :
                r1 = int(row[-1][row[-1].index('(')+1])
                r1 = functions[0](r1)
            elif delims[1] in row[-2] :
                r2 = int(row[-1][row[-1].index('(')+1])
                r2 = functions[1](r2)
            elif delims[2] in row[-2] :
                r3 = int(row[-1][row[-1].index('(')+1])
                r3 = functions[2](r3)

        if r1 == None or r2 == None or r3 == None :
            continue
        
        next_line = line
        line = cur_data[-1]
        line = line.split(',')

        # remove . for strptime
        if '.' in line[6] :
            line[6] = line[6][:line[6].index('.')]
        if '.' in line[7] :
            line[7] = line[7][:line[7].index('.')]
        if '.' in line[4] :
            line[4] = line[4][:line[4].index('.')]
        if '.' in line[5] :
            line[5] = line[5][:line[5].index('.')]
        
        # first person
        if last_id == None :
            last_id = line[2]
            first_time = datetime.strptime(line[6] + ' ' + line[7], date_format)
        
        # new person
        if last_id != line[2] :
            cur_id += 1
            first_time = datetime.strptime(line[6] + ' ' + line[7], date_format)
            last_id = line[2]
            day_dict = {}
            total_days = 0

        # get times
        start_time = datetime.strptime(line[4] + ' ' + line[5], date_format)
        cur_time = datetime.strptime(line[6] + ' ' + line[7], date_format)
        cur_min = (cur_time.hour*60) + cur_time.minute
        
        cur_hour = cur_time.hour

        # get current day (0=start)
        if day_dict.get(cur_time.day) == None :
            day_dict[cur_time.day] = total_days
            total_days += 1

        cur_day = day_dict[cur_time.day] 

        # get time since study started and response time
        delta = int((cur_time - first_time).total_seconds() / 60)
        response_time = (cur_time - start_time).total_seconds()

        data.append([id + cur_id, delta, cur_min,cur_hour,cur_day,int(response_time),r1, r2, r3])

        line = next_line
    file.close()
    return data

def in_delim(delims, str) :
    for delim in delims :
        if delim in str :
            return True
    return False

def verify_better(path, delims,triggers=None) :
    
    # open file, skip header
    file = open(path, 'r')
    file.readline()

    # keeps tracks of which day it is
    day_dict = {}
    total_days = 0

    users = {}

    line = file.readline()
    cur_data = []

    # prompt errors, total
    prompt_errors = 0
    total_prompts = 0

    # question errors, total
    question_errors = 0
    total_questions = 0

    # sharp/stress/fatigue errors, total
    cog_errors = 0
    cog_question_total = 0

    # 30 days of data
    hourly_times = [0] * 24
    adjusted_day_times = [0] * 420
    all_times = [0] * 720

    # 30 days of data
    hourly_times_responded = [0] * 24
    adjusted_day_times_responded = [0] * 420
    all_times_responded = [0] * 720

    # 30 days of data
    cog_hourly_times = [0] * 24
    cog_adjusted_day_times = [0] * 420
    cog_all_times = [0] * 720

    # 30 days of data
    cog_hourly_times_responded = [0] * 24
    cog_adjusted_day_times_responded = [0] * 420
    cog_all_times_responded = [0] * 720

    question_freqs = {}
    error_freqs = {}

    single_prompts = [0,0]
    big_prompts = [0,0]

    fail_lengths = [0] * 20
    total_lengths = [0] * 20

    prompt_type = [0,0]
    prompt_successes = [0,0]

    # goes through each line
    fc=0
    while line != '':
        if total_questions > 17000 :
            s = 1
        prompt = line[:line.index(',')]
        old_prompt = prompt
        cur_data = []

        # gets current prompt values
        while prompt == old_prompt and line != '' :
            old_prompt = prompt
            COMMA_MATCHER = re.compile(r",(?=(?:[^\"']*[\"'][^\"']*[\"'])*[^\"']*$)")
            split_result = COMMA_MATCHER.split(line)
            split_result[-2] = split_result[-2].replace('\"', '')
            cur_data.append(split_result)
            line = file.readline()
            if line != '' :
                prompt = line[:line.index(',')]
        
        error_found = False
        if cur_data[0][3] == 'random' :
            prompt_type[0] += 1
        else :
            prompt_type[1] += 1

        if triggers != None and cur_data[0][3] not in triggers:
            continue
        
        for row in cur_data :
            # new user, reset
            if users.get(row[2]) == None :
                users[row[2]] = 1
                day_dict = {}
                total_days = 0

            # removes . from time for strptime
            # gets time
            time = row[4] + ' ' + row[5]
            time = time.replace('.', '')
            time = datetime.strptime(time, "%Y-%m-%d %H:%M:%S")

            # new day
            if day_dict.get(time.day) == None :
                day_dict[time.day] = total_days
                total_days += 1


            if question_freqs.get(row[-2]) == None :
                question_freqs[row[-2]] = 0
            question_freqs[row[-2]] += 1 
            
            if (row[-1] == '\n' or row[-1] == '') and not filter_line(cur_data, row) :
                day = day_dict[time.day] 

                # get hour on 8-21 scale
                t = time.hour #+ (day*24)
                t2 = t + (day*14)

                hourly_times[t] += 1

                # if during day
                if t >=8 and t < 22 :
                    adjusted_day_times[t2-8] += 1
                
                all_times[t + (day*24)] += 1

                if not error_found :
                    fail_lengths[len(cur_data)-1] += 1
                    if len(cur_data) < 4:
                        single_prompts[1] += 1
                    else :
                        big_prompts[1] += 1
                    prompt_errors += 1
                    error_found = True
                question_errors += 1

                if error_freqs.get(row[-2]) == None :
                    error_freqs[row[-2]] = 0
                error_freqs[row[-2]] += 1 
                
                if in_delim(delims, row[-2]):
                    cog_errors += 1
                    cog_hourly_times[t] += 1

                    # if during day
                    if t >=8 and t < 22 :
                        cog_adjusted_day_times[t2-8] += 1
                    
                    cog_all_times[t + (day*24)] += 1
            else :
                if filter_line(cur_data, row) :
                    fc+=1
                day = day_dict[time.day] 

                # get hour on 8-21 scale
                t = time.hour #+ (day*24)
                t2 = t + (day*14)

                hourly_times_responded[t] += 1

                # if during day
                if t >=8 and t < 22 :
                    adjusted_day_times_responded[t2-8] += 1
                
                all_times_responded[t + (day*24)] += 1
                if in_delim(delims, row[-2]) :
                    cog_question_total += 1
                    cog_hourly_times_responded[t] += 1

                    # if during day
                    if t >=8 and t < 22 :
                        cog_adjusted_day_times_responded[t2-8] += 1
                    
                    cog_all_times_responded[t + (day*24)] += 1

            total_questions += 1
            if in_delim(delims, row[-2]) :
                cog_question_total += 1

        if not error_found :
            if cur_data[0][3] == 'random' :
                prompt_successes[0] += 1
            else :
                prompt_successes[1] += 1
        if len(cur_data) < 4 :
            single_prompts[0] += 1
        else :
            big_prompts[0] += 1
        total_prompts += 1
        total_lengths[len(cur_data) -1] += 1
    
    i = len(adjusted_day_times)-1
    if question_errors == 0 :
        print("NOTHING WRONG")
        return
    # remove all extra empty days
    while adjusted_day_times[i] == 0:
        del adjusted_day_times[i]
        i-=1
    i = len(adjusted_day_times_responded)-1
    while adjusted_day_times_responded[i] == 0 :
        del adjusted_day_times_responded[i]
        i -= 1
    
    i = len(all_times)-1
    while all_times[i] == 0 :
        del all_times_responded[i]
        del all_times[i]
        i -=1
    
    # # graph all missed times 8:00-21:00
    # plt.bar(np.arange(0,len(all_times)), all_times,color='r')
    # plt.bar(np.arange(0, len(all_times_responded)), all_times_responded,bottom=all_times,color='b')
    # plt.title("Response Times over Study")
    # plt.legend(["Missed", "Responded"])
    # plt.xlabel("Time (Days)")
    # ticks = [i for i in range(len(all_times)) if i % 24 == 0] # for use in entire study graph
    # plt.xticks(ticks, np.array(ticks) // 24)
    # plt.ylabel("Number Questions")
    # plt.show()

    # # graph all missed times over day
    # plt.bar(np.arange(0,len(hourly_times)), hourly_times,color='r')
    # plt.bar(np.arange(0, len(hourly_times_responded)), hourly_times_responded,bottom=hourly_times,color='b')
    # plt.legend(["Missed", "Responded"])
    # plt.title("Response Times over Day")
    # plt.xlabel("Time (Days)")
    # plt.xticks([0, 3, 6, 9, 12, 15, 18, 21, 24], ['12 am', '3 am', '6 am', '9 am', '12 pm', '3 pm', '6 pm', '9 pm','12 am'])
    # #ticks = [i for i in range(len(all_times)) if i % 24 == 0] # for use in entire study graph
    # #plt.xticks(ticks, np.array(ticks) // 24)
    # plt.ylabel("Number Questions")
    # plt.show()

    # graph all missed times 8:00-21:00
    plt.bar(np.arange(0,len(cog_all_times)), cog_all_times,color='r')
    plt.bar(np.arange(0, len(cog_all_times_responded)), cog_all_times_responded,bottom=cog_all_times,color='b')
    plt.title("Response Times over Study")
    plt.legend(["Missed", "Responded"])
    plt.xlabel("Time (Days)")
    ticks = [i for i in range(len(all_times)) if i % 24 == 0] # for use in entire study graph
    plt.xticks(ticks, np.array(ticks) // 24)
    plt.ylabel("Number Questions")
    plt.show()

    # # graph all missed times over day
    # plt.bar(np.arange(0,len(cog_hourly_times)), cog_hourly_times,color='r')
    # plt.bar(np.arange(0, len(cog_hourly_times_responded)), cog_hourly_times_responded,bottom=cog_hourly_times,color='b')
    # plt.legend(["Missed", "Responded"])
    # plt.title("Response Times over Day (EMA)")
    # plt.xlabel("Time (Days)")
    # plt.xticks([0, 3, 6, 9, 12, 15, 18, 21, 24], ['12 am', '3 am', '6 am', '9 am', '12 pm', '3 pm', '6 pm', '9 pm','12 am'])
    # #ticks = [i for i in range(len(all_times)) if i % 24 == 0] # for use in entire study graph
    # #plt.xticks(ticks, np.array(ticks) // 24)
    # plt.ylabel("Number Questions")
    # plt.show()
    
    # get 8-21 data points, make features
    hourly_times = hourly_times[8:21]
    cur = make_features(hourly_times)
    print_data(cur, "Hourly")

    # make features across whole study 8-21
    cur = make_features(adjusted_day_times)
    print_data(cur, "Study")

    # get 8-21 data points, make features
    hourly_times_responded = hourly_times_responded[8:21]
    cur = make_features(hourly_times_responded)
    print_data(cur, "Hourly Success")

    # make features across whole study 8-21
    cur = make_features(adjusted_day_times_responded)
    print_data(cur, "Study Success")

    cog_ratio = round(cog_errors/cog_question_total, 2)
    all_ratio = round(question_errors/total_questions, 2)
    prompt_ratio = round(prompt_errors/total_prompts,2)

    # print question followed by count
    print('-----------')
    print("ALL QUESTIONS")
    for k,v in question_freqs.items() :
        failures = 0
        if error_freqs.get(k) != None :
            failures = error_freqs[k]
        print(f'{k}   {failures}/{v}')
    print('-----------')

    print(f"Path:{path}, cog error, all error: {cog_errors}/{cog_question_total} {question_errors}/{total_questions}, cog ratio: {cog_ratio}, all ratio: {all_ratio}")
    print(f"Prompt error: {prompt_errors}/{total_prompts}, ratio: {prompt_ratio}")
    print(f"Prompt sizes: small {single_prompts[1]}/{single_prompts[0]}, big: {big_prompts[1]}/{big_prompts[0]}")
    print(fail_lengths)
    print(total_lengths)
    print(prompt_type, (prompt_type[0]/(prompt_type[0] + prompt_type[1])))
    print(prompt_successes, (prompt_successes[0]/(prompt_successes[0] + prompt_successes[1])))

def check_lines(lines, delim, threshold) :
    for l in lines :
        if delim in l[-2] and l[-1] != '\n':
            try :
                value = int(l[-1][1])
                return threshold(value)
            except :
                return threshold(l[-1])
    return False

def filter_fau(lines, line) :
    #ucd: no filters
    threshold = lambda x : x >= 0
    if 'motivated' in line[-2] or 'environment' in line[-2]:
        return check_lines(lines, 'anxious', threshold)
    return False

def filter_dyad(lines, line) :
    threshold = lambda x: x[x.index('(')+1:x.index(')')] == "None"
    if 'angry' in line[-2] :
        return check_lines(lines, 'interaction have', threshold)
    return False

def filter_gsur(lines, line) :
    threshold = lambda x : x < 2
    if 'which environ-' in line[-2] :
        return check_lines(lines, 'context', threshold) 
    elif 'internal factor' in line[-2]:
        return check_lines(lines, 'internal state', threshold) 

    return False

def filter_rwcs(lines, line) :
    threshold = lambda x: x == 1

    if 'most influential enviro' in line[-2] :
        return check_lines(lines, 'my environment is', threshold)
    elif 'influential internal factor' in line[-2] :
        return check_lines(lines, 'my internal state is', threshold)
    elif 'I used the strategy to' in line[-2] :
        return check_lines(lines, 'the most helpful strategy', threshold)
    
    return False
                
def filter_dod(lines, line) :
    threshold = lambda x : x == 1
    if 'distracting environ-' in line[-2] :
        return check_lines(lines, 'my environment is distracting', threshold)
    elif 'most distracting internal factor' in line[-2] :
        return check_lines(lines, 'my internal self is distracting', threshold)
    elif 'most influential environ-' in line[-2] :
        return check_lines(lines, 'my environment is influencing', threshold)
    elif 'most influential internal factor' in line[-2] :
        return check_lines(lines, 'my internal self is influencing', threshold)
    
    return False


def filter_line(lines, line) :
    if 'emma' in line[2] :
        return filter_dod(lines ,line)
    elif 'dyad' in line[2] :
        return filter_dyad(lines ,line)
    elif 'fau' in line[2] :
        return filter_fau(lines, line)
    elif 'luna' in line[2] :
        return filter_gsur(lines, line)
    elif 'rwcs' in line[2] :
        return filter_rwcs(lines, line)

folder = "./ema_nback/"
folder2 = "./data/"

# functions to apply to the ema numbers to convert into range of [1,5] with 1 being low stress/fatigue/sharp
six = lambda x : 6 - x
same = lambda x : x
rnd = lambda x : int(round(x*(5.0/7.0)))
six_rnd = lambda x : 6 - rnd(x)
six_add = lambda x : 6 - add(x)
add = lambda x : x + 1

default_format = "%Y-%m-%d %H:%M:%S"
dod_format = "%m/%d/%Y %H:%M:%S"
default_functions = [same,six,six]
default_delims = ["sharp", "fatigue", "stress"]

#quickcheck(folder2 + 'rwcs_ema.csv')

normal_triggers = ['random', 'after_proximity']
# verify_better('./mega.csv', ["sharp", "fatigue", "anxious", "independent", "over-", "motivated", "stress", "confident", "going"],triggers=normal_triggers) #

# verify_better(folder2 + 'dod_ema.csv', ["sharp", "independent", "over"],triggers=normal_triggers)
# verify_better(folder2 + 'dyad_ema.csv', ["confident", "going", "anxious"],triggers=normal_triggers)
# verify_better(folder2 + 'fau_ema.csv', ["sharp", "fatigue", "anxious"],triggers=normal_triggers)
# verify_better(folder2 + 'func_ema.csv', ["motivated", "fatigue", "stress"],triggers=normal_triggers)
# verify_better(folder2 + 'gsur1_ema.csv', default_delims,triggers=normal_triggers)
# verify_better(folder2 + 'gsur2_ema.csv', ["sharp", "fatigue"],triggers=normal_triggers)
# verify_better(folder2 + 'rwcs_ema.csv', default_delims,triggers=normal_triggers)
# verify_better(folder2 + 'ucd_ema.csv', default_delims,triggers=normal_triggers)

# rwcs = read_prompts(folder + "rwcs_ema.csv", 8, ["sharp", "fatigue", "stress"], default_format)
# gsur_2 = read_prompts(folder + "gsur_2.0_ema.csv", 6, ["sharp", "fatigue", "stress"], default_format)

chile = read_data(folder + "chile_ema.csv", 1, default_delims, default_format,[rnd,six_rnd,six_rnd]) 
dod = read_data(folder + "dod_ema.csv", 2, ["sharp", "independent", "over"],dod_format,[rnd,rnd,six_rnd])   
dyad = read_data(folder + "dyad_ema.csv", 3, ["confident", "going", "anxious"], default_format,[add,add,six_add])    
fau = read_data(folder + "fau_ema.csv", 4, ["sharp", "fatigue", "anxious"],default_format,default_functions)   
func =  read_data(folder + "func_ema.csv", 5, ["motivated", "fatigue", "stress"],default_format,default_functions)   
gsur = read_data(folder + "gsur_ema.csv", 6, default_delims,default_format,default_functions)   
ucd = read_data(folder + "ucd_ema.csv", 7, default_delims, default_format,default_functions)   
rwcs = read_data(folder + "rwcs_ema.csv", 8, default_delims, default_format,default_functions) 

ihs = read_ihs(folder + "ihs_ema.csv", 9)

all_data = chile + dod + dyad + fau + func + gsur + ucd + rwcs #+ ihs

file = open('ema_new.csv', 'w',newline="")

writer = csv.writer(file)
writer.writerow(["id", "time_elapsed", "minute", "hour", "day","response_time","sharp","fatigue","stress"])
writer.writerows(all_data)
file.close()