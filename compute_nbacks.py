import csv
from datetime import datetime
def read_data(path, id) :

    # allows for 1000 ids
    id *= 1000
    file = open(path, 'r')

    reader = csv.reader(file)

    date_format = "%Y-%m-%d %H:%M:%S"

    cur_id = 0
    last_id = None
    data = []

    day_dict = {}
    total_days = 0
    first_time = None

    for row in reader :
        if last_id == None :
            last_id = row[0]
            first_time = datetime.strptime(row[1] + ' ' + row[2], date_format)
        
        if last_id != row[0] :
            cur_id += 1
            first_time = datetime.strptime(row[1] + ' ' + row[2], date_format)
            last_id = row[0]
            day_dict = {}
            total_days = 0

        cur_time = datetime.strptime(row[1] + ' ' + row[2], date_format)
        cur_min = (cur_time.hour*60) + cur_time.minute
        
        cur_hour = cur_time.hour
        if day_dict.get(cur_time.day) == None :
            day_dict[cur_time.day] = total_days
            total_days += 1
        
        cur_day = day_dict[cur_time.day] 

        delta = int((cur_time - first_time).total_seconds() / 60)
        try :
            data.append([id + cur_id, delta, cur_min,cur_hour,cur_day, int(row[3])])
        except :
            pass

    return data

def read_dod(path, id) :
    id *= 1000
    file = open(path, 'r')
    file.readline()
    reader = csv.reader(file)

    date_format = "%Y-%m-%d %H:%M:%S"

    cur_id = 0
    last_id = None
    data = []


    day_dict = {}
    total_days = 0
    first_time = None

    for row in reader :
        if last_id == None :
            last_id = row[2]
            first_time = datetime.strptime(row[4] + ' ' + row[5], date_format)
        
        if last_id != row[2] :
            cur_id += 1
            first_time = datetime.strptime(row[4] + ' ' + row[5], date_format)
            last_id = row[2]
            day_dict = {}
            total_days = 0

        cur_time = datetime.strptime(row[4] + ' ' + row[5], date_format)
        cur_min = (cur_time.hour*60) + cur_time.minute
        
        cur_hour = cur_time.hour
        if day_dict.get(cur_time.day) == None :
            day_dict[cur_time.day] = total_days
            total_days += 1
        
        cur_day = day_dict[cur_time.day] 

        delta = int((cur_time - first_time).total_seconds() / 60)
        try :
            data.append([id + cur_id, delta, cur_min,cur_hour,cur_day, int(row[6])])
        except :
            pass

    return data

def main() :
    folder = "./ema_nback/"
    chile = read_data(folder + "chile_nback.csv",1)
    dod = read_dod(folder + "dod_nback.csv",2)
    fau = read_data(folder + "fau_nback.csv",4)
    func = read_data(folder + "func_nback.csv",5)
    gsur = read_data(folder + "gsur_nback.csv",6)
    ucd = read_data(folder + "ucd_nback.csv",7)
    #ihs = read_data(folder + "ihs_nback.csv",8) not used because has missing time values
    

    all_data = chile + dod + fau + func + gsur + ucd #+ ihs

    file = open('nback.csv', 'w',newline="")

    writer = csv.writer(file)
    writer.writerow(["id", "time_elapsed", "minute", "hour", "day","score"])
    writer.writerows(all_data)
    file.close()

if __name__ == '__main__' :
    main()