from datetime import datetime
import csv
import matplotlib.pyplot as plt
import numpy as np
from read_and_graph_data import make_features, print_data

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

            # checks if has correct word, then checks if not empty
            if delims[0] in row[-2] and row[-1] != '\n':
                r1 = int(row[-1][row[-1].index('(')+1])
                r1 = functions[0](r1)
            elif delims[1] in row[-2] and row[-1] != '\n':
                r2 = int(row[-1][row[-1].index('(')+1])
                r2 = functions[1](r2)
            elif delims[2] in row[-2] and row[-1] != '\n':
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

def verify_emas(path, delims) :
    # for checking totals across all studies
    global linetot
    global errortot

    file = open(path, 'r')
    file.readline()

    reader = csv.reader(file)

    # count of errors of ssf
    cognitive_errors = 0

    # count of all errors
    all_errors = 0

    # count of ssf questions
    cog_question_count = 0

    # count of all questions
    all_question_count = 0

    # counts of errors in each prompt
    errors_count_dict = {}

    # size of each prompt
    prompt_sizes = {}

    # all users
    users = {}

    # count per question
    questions_count = {}

    # error per question
    question_errors = {}

    # 20 days of data
    hourly_times = [0] * 24
    adjusted_day_times = [0] * 280
    all_times = [0] * 480 

    day_dict = {}
    total_days = 0

    for row in reader :
        st = ''.join(row)

        # count number of questions in prompt
        if prompt_sizes.get(row[0]) == None :
            prompt_sizes[row[0]] = 0
        
        prompt_sizes[row[0]] += 1

        # removes . from time for strptime
        if '.' in row[6] :
            row[6] = row[6][:row[6].index('.')]
        if '.' in row[7] :
            row[7] = row[7][:row[7].index('.')]

        # gets time
        time = row[6] + ' ' + row[7]
        time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')

        # new user, reset
        if users.get(row[2]) == None :
            users[row[2]] = 1
            day_dict = {}
            total_days = 0

        # new day
        if day_dict.get(time.day) == None :
            day_dict[time.day] = total_days
            total_days += 1

        # count number of times each question appears
        if questions_count.get(row[-2]) == None :
            questions_count[row[-2]] = 0
        questions_count[row[-2]] +=1

        # if no response
        if row[-1] == '' :
            # count number of times each question appears
            if question_errors.get(row[-2]) == None :
                question_errors[row[-2]] = 0
            question_errors[row[-2]] +=1
            # count errors in prompt
            if errors_count_dict.get(row[0]) == None :
                errors_count_dict[row[0]] = 0
            
            errors_count_dict[row[0]] += 1

            day = day_dict[time.day] 

            # get hour on 8-21 scale
            t = time.hour #+ (day*24)
            t2 = t + (day*14)

            hourly_times[t] += 1

            # if during day
            if t >=8 and t < 22 :
                adjusted_day_times[t2-8] += 1
            
            all_times[t + (day*24)] += 1
            
            # if a s,s,f question
            if delims[0] in st or delims[1] in st or delims[2] in st :
                cognitive_errors += 1
                errortot += 1
            all_errors += 1

        # checks if ssf question
        if delims[0] in st or delims[1] in st or delims[2] in st :
            cog_question_count += 1
        all_question_count += 1
        linetot += 1
    
    i = len(adjusted_day_times)-1

    # remove all extra empty days
    while adjusted_day_times[i] == 0:
        del adjusted_day_times[i]
        i-=1

    # graph all missed times 8:00-21:00
    plt.bar(np.arange(0,len(adjusted_day_times)), adjusted_day_times)
    plt.show()

    # graph all missed times over day
    plt.bar(np.arange(0,len(hourly_times)), hourly_times)
   
    plt.title("Response Failure Times over Study")
    plt.xlabel("Time (Days)")
    plt.xticks([0, 3, 6, 9, 12, 15, 18, 21, 24], ['12 am', '3 am', '6 am', '9 am', '12 pm', '3 pm', '6 pm', '9 pm','12 am'])
    #ticks = [i for i in range(len(times3)) if i % 24 == 0] # for use in entire study graph
    #plt.xticks(ticks, np.array(ticks) // 24)
    plt.ylabel("Prompt Failures")
    plt.show()
    
    # get 8-21 data points, make features
    hourly_times = hourly_times[8:21]
    cur = make_features(hourly_times)
    print_data(cur, "Times")

    # make features across whole study 8-21
    cur = make_features(adjusted_day_times)
    print_data(cur, "Study")


    prompt_size_counts = [0] * 20
    error_counts = [0] * 20

    # print question followed by count
    print('-----------')
    print("ALL QUESTIONS")
    for k,v in questions_count.items() :
        print(f'{k} {v}')
    print('-----------')

    # print failed question followed by count
    print("FAILED QUESTIONS")
    for k,v in question_errors.items() :
        print(f'{k} {v}')
    print('-----------')
    
    # goes through each length of prompt, counts how many time each length failed
    for key, value in prompt_sizes.items() :
        if errors_count_dict.get(key) != None :
            prompt_size_counts[value-1] += 1 

    # goes through all errors, 
    for key, value in errors_count_dict.items() :
        error_counts[value-1] += 1 

    plt.bar(np.arange(1,len(prompt_size_counts)+1), prompt_size_counts)
   
    plt.title("Prompt Length vs Amount of Failures")
    plt.xlabel("Prompt Length")
    plt.ylabel("Prompt Failures")
    plt.xticks([1,2,3,4,5,6,7,8,9])
    plt.show()
    print("prompt lengths:")
    print(prompt_size_counts)
    print("prompt fails length:")
    print(error_counts)
    print(f"Path:{path}, errors: {cognitive_errors} {all_errors}, lines: {cog_question_count} {all_question_count}, ratio: {cognitive_errors/cog_question_count}, ratio2: {all_errors/all_question_count}")

    file.close()


folder = "./ema_nback/"

# generates the response rate statistics
verify_emas(folder + "rwcs_ema.csv", ["sharp", "fatigue", "stress"])
#verify_emas(folder + "gsur_2.0_ema.csv", ["sharp", "fatigue", "stress"])




# functions to apply to the ema numbers to convert into range of [1,5] with 1 being low stress/fatigue/sharp
six = lambda x : 6 - x
same = lambda x : x
rnd = lambda x : int(round(x*(5.0/7.0)))
six_rnd = lambda x : 6 - rnd(x)
six_add = lambda x : 6 - add(x)
add = lambda x : x + 1

default_format = "%Y-%m-%d %H:%M:%S"
dod_format = "%m/%d/%Y %H:%M:%S"

# rwcs = read_prompts(folder + "rwcs_ema.csv", 8, ["sharp", "fatigue", "stress"], default_format)
# gsur_2 = read_prompts(folder + "gsur_2.0_ema.csv", 6, ["sharp", "fatigue", "stress"], default_format)

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