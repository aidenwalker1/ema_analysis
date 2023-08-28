import csv
import numpy as np
import matplotlib.pyplot as plt
import tsfresh.feature_extraction.feature_calculators as tcalc
import stat_features
from scipy.stats import linregress
import scipy.signal

day_index = lambda x : x[3]
total_index = lambda x: x[3] + (x[4]*24)

def read_ema(path:str) -> list:
    """
        Reads EMA file into list
    
        Parameters
        ----------
        path : str
            Path to csv file
    
        Returns
        -------
        list
            All ema data in form (study,person,response)
    """

    # opens file, skips first line
    file = open(path, 'r')
    file.readline()

    reader = csv.reader(file)

    all_data = []
    id_data = []
    study_data = []

    # previous id to check when person changes
    old_id = -1

    # goes through all lines
    for line in reader :
        id = line[0]

        # compares ids
        if id != old_id :
            # checks if not first person
            if old_id != -1 :
                # checks if study is different
                if old_id[0] != id[0] :
                    all_data.append(study_data)
                    study_data = []
                
                # adds new person to study if more than 4 responses
                if len(id_data) > 4 :
                    study_data.append(id_data)
                id_data = []
            old_id = id
        
        # get data, save aggregate of responses
        newdata = [int(l) for l in line]
        newdata.append((newdata[6] + newdata[7] + newdata[8] ) / 3)

        id_data.append(newdata)

    #appends final study
    if old_id != -1 :
        all_data.append(study_data)

    file.close()

    return all_data

def read_nback(path:str) -> list:
    """
        Reads nback file into list
    
        Parameters
        ----------
        path : str
            Path to csv file
    
        Returns
        -------
        list
            All nback data in form (study,person,response)
    """

    # opens file, skips first line
    file = open(path, 'r')
    file.readline()

    reader = csv.reader(file)

    all_data = []
    id_data = []
    study_data = []

    # previous id to check when person changes
    old_id = -1

    # goes through all lines
    for line in reader :
        id = line[0]

        # compares ids
        if id != old_id :
            # checks if not first person
            if old_id != -1 :
                # checks if study is different
                if old_id[0] != id[0] :
                    all_data.append(study_data)
                    study_data = []
                
                # adds new person to study if more than 4 responses
                if len(id_data) > 4 :
                    study_data.append(id_data)
                id_data = []
            old_id = id
        
        # get data, save aggregate of responses
        newdata = [int(l) for l in line]

        id_data.append(newdata)

    #appends final study
    if old_id != -1 :
        all_data.append(study_data)

    file.close()

    return all_data

def flatten(data:list, start:int, end:int, cur=0) -> list:
    """
        Reads EMA file into list
    
        Parameters
        ----------
        data : list
            list of data to flatten
        start : int
            Start index to flatten
        end : int
            End index to flatten 

        Returns
        -------
        list
            List flattened in [start,end)
    """

    # checks if at end index, before start, or in middle
    if cur == end :
        return [data]
    elif cur < start :
        # goes to next layer to flatten
        for i in range(len(data)) :
            data[i] = flatten(data[i], start, end, cur + 1)
    else:
        flat = []
        
        # flatten layer 
        for lines in data :
            flat += flatten(lines, start, end, cur + 1)
            
        return flat
    
    return data

def make_features(x:list) -> list:
    """
        Creates statistical features from a list
    
        Parameters
        ----------
        x : list
            list of data

        Returns
        -------
        list
            List of features
    """

    x = np.array(x)

    # gets mean and std
    avg = np.mean(x)
    std = np.std(x)

    # length of input
    length = x.shape[0]

    # difference of start to end
    diff = np.mean(x[(3*length)//4:]) - np.mean(x[:length//4])
    diff = np.nan_to_num(diff)

    # absolute difference
    absdiff = abs(diff)
    times = [i for i in range(x.shape[0])]

    # slope of regression
    slope = linregress(times, x).slope
    slope = np.nan_to_num(slope)

    # min, max
    max_val = np.max(x)
    min_val = np.min(x)

    # range
    rg = max_val - min_val

    # frequency of items
    freqs = {}
    for val in x :
       value = int(round(val))-1
       if freqs.get(value) == None :
           freqs[value] = 0
       freqs[value] += 1

    # quartiles
    q1 = tcalc.quantile(x, 0.25)
    q3 = tcalc.quantile(x, 0.75)

    # skew, kurtois, autocorrelation
    skew = tcalc.skewness(x)
    kurt = tcalc.kurtosis(x)
    ac = 0

    if length > 1:
        ac = stat_features.autocorrelation(x, avg)
    
    # calculate PSD
    f, p = scipy.signal.welch(x)
    
    # sort psd
    most = sorted(p,reverse=True)

    m = []

    # gets highest frequencies
    for val in most :
        if val / most[0] < 0.75 :
            break
        m.append(val)
    
    # compares highest frequencies to mean
    fft = np.mean(m) / np.mean(p)
    fft = np.nan_to_num(fft)

    return [avg,slope,diff,fft,ac,std,absdiff, max_val,min_val,rg,q1,q3,skew,kurt] #+ list(freqs.items())

def generate_user_data(user:list,size:int,index_func,type:str) -> list:
    """
        Combines user data into single day
    
        Parameters
        ----------
        user : list
            list of user responses

        Returns
        -------
        list
            List user responses across 1 day
    """
    if type.lower() == "ema" :
        selection = 6
        total_elems = 4
    else :
        selection = 5
        total_elems = 1
    data = []

    # append size lists
    for i in range(size) :
        data.append([])

    # go through each response
    for i in range(len(user)-1):
        # get current, next response
        cur = user[i]
        next_resp = user[i+1]
        cur_index = index_func(cur)
        next_index = index_func(next_resp)
        # save current selection at current hour
        data[cur_index].append(cur[selection:selection+total_elems])

        # if same day, fill in gaps
        if cur[4] == next_resp[4] :
            # gets average
            avg = (np.array(cur[-1]) + np.array(next_resp[selection:selection+total_elems])) / 2
            avg = avg.tolist()

            # fills in each unresponded data point
            for j in range(cur_index, next_index) :
                data[j].append(avg)

    # adds final data point
    data[user[-1][3]].append(user[-1][selection:selection+total_elems])

    return data

def print_data(data,title) :
    """
        Prints data
    
        Parameters
        ----------
        user : list
            list of user responses

        Returns
        -------
        list
            List user responses across 1 day
    """
    print(f'\n{title}:\nAverage: {data[0]:.3f}\nStd: {data[5]:.3f}\nDifference: {data[2]:.3f}\nAbsolute Difference: {data[6]:.3f}\nFFT Ratio: {data[3]:.3f}\nAutocorrelation: {data[4]:.3f}\nSlope: {data[1]:.3f}\n' )

def create_data(user_data:list,time_type:str,type:str) -> (list,list,list,list,list,list):
    """
        Creates the data and stats from the list of user data
    
        Parameters
        ----------
        user_data : list
            list of user responses
        time_type: str {'day', 'study'}
            type of time series
        type: str
            ema or nback

        Returns
        -------
        tuple
            averaged data, averaged data without empty timesteps, std, std without empty timesteps, computed user stats, raw user data
    """
    index_func = day_index
    size = 24
    if time_type.lower() == 'study' :
        index_func = total_index
        size = get_size(user_data)

    # combined user data
    combined_data = [[] for i in range(size)]

    # computed stats for each user
    user_computed = []

    # raw data of each user, averaged at each time step if more than 1 response
    user_raw_data = []

    # goes through every user
    for user in user_data :
        # generates data for user across each day
        day_data = generate_user_data(user,size, index_func,type=type)

        for i, line in enumerate(day_data) :
            combined_data[i] += line

        # compresses each response at a timestep to average, saves the 0s
        averaged = [np.average(l,axis=0).tolist() if len(l) > 0 else [0,0,0,0] for l in day_data]

        # compresses the data, removing times with no responses
        averaged_zeroless = [np.average(l,axis=0) for l in day_data if len(l) > 0]

        # generates features across data, using combined response
        if type == "ema" :
            user_computed.append(make_features([c[3] for c in averaged_zeroless]))
        else :
            user_computed.append(make_features([c[0] for c in averaged_zeroless]))
        
        user_raw_data.append(averaged)
    
    # compresses all data to get average at each step
    total = [np.average(l,axis=0).tolist() if len(l) > 4 else [0,0,0,0] for l in combined_data]

    # compresses all data with responses for analysis later
    total_zeroless = [np.average(l,axis=0).tolist() for l in combined_data if len(l) > 4]

    # does the same, but with the std at each step
    tot_stds = [np.std(l,axis=0).tolist() if len(l) > 4 else [0,0,0,0] for l in combined_data]
    tot_stds_zeroless = [np.std(l,axis=0).tolist() for l in combined_data if len(l) > 4]

    return (total, total_zeroless, tot_stds, tot_stds_zeroless, user_computed, user_raw_data)

def analyze_times(responses:list, type:str) :
    """
        Analyzes response times
    
        Parameters
        ----------
        responses : list
            list of user responses
        size : int
            size of time series     

        Returns
        -------
        list
            averaged data, averaged data without empty timesteps, std, std without empty timesteps, computed user data, raw user data
    """

    size = 24
    index_func = day_index
    if type.lower() == 'study' :
        size = get_size(responses)
        index_func = total_index

    # list of number of responses at each time step
    data = [0] * size

    # goes through each response
    for response in responses :
        # gets time, then adds 1 to index
        index = index_func(response)

        data[index] += 1
    
    # show graph
    plt.bar(np.arange(len(data)),data)
    if type == "Day" : 
        plt.title("Responses Per Hour")
        plt.ylabel("Total Responses")
        plt.xlabel("Time of Day")
        plt.xticks([0, 3, 6, 9, 12, 15, 18, 21, 24], ['12 am', '3 am', '6 am', '9 am', '12 pm', '3 pm', '6 pm', '9 pm','12 am'])
    else :
        ticks = np.array([i for i in range(len(data)) if i % 24 == 0])
        plt.title("Responses Per Hour")
        plt.ylabel("Total Responses")
        plt.xlabel("Time (Day)")
        plt.xticks(ticks, ticks // 24)
    plt.show()

def analyze_nback(packed_data:tuple,type:str) :
    """
        Analyzes nback responses
    
        Parameters
        ----------
        packed_data : tuple
            6 lists created by create_data
        type: str ["Day", "Study"]
            type of data being analyzed       

    """
    
    # unpacks data
    average, stats, std, std_stats, user_stats, user_raw_data = packed_data

    convert = lambda x,i : [t[i] for t in x]

    #user_raw_data = np.array(user_raw_data)

    # plots each line
    for line in user_raw_data :
        plt.plot(convert(line,0),alpha=0.1)

    # plots averaged graph
    plt.plot(convert(average,0), 'r')

    if type.lower() == "day" :
        plt.title("Average Participant Score Over Day")
        plt.xticks([0, 3, 6, 9, 12, 15, 18, 21, 24], ['12 am', '3 am', '6 am', '9 am', '12 pm', '3 pm', '6 pm', '9 pm','12 am'])
        plt.xlabel("Time of Day")
    else :
        ticks = np.array([i for i in range(len(convert(average,0))) if i % 24 == 0])
        plt.title("Average Participant Score Over Study")
        plt.xticks(ticks, ticks // 24)
        plt.xlabel("Time (1 Day)")
    plt.show()

    # prints stats for population, std population, and user data
    new_stats = make_features(convert(stats,0))
    new_std_stats = make_features(convert(std_stats,0))
    new_user_stats = np.average(user_stats,axis=0)

    print_data(new_stats, "Population")
    print_data(new_std_stats, "Std")
    print_data(new_user_stats, "Users")

    # prints graph score
    for i in range(0,1) :
        compressed = convert(average,i)

        plt.plot(compressed, 'r')
        if type.lower() == 'day' :
            plt.title("Average Participant Score Over Day")
            plt.xticks([0, 3, 6, 9, 12, 15, 18, 21, 24], ['12 am', '3 am', '6 am', '9 am', '12 pm', '3 pm', '6 pm', '9 pm','12 am'])
            plt.xlabel("Time of Day")
        else :
            ticks = np.array([i for i in range(len(compressed)) if i % 24 == 0])
            plt.title("Average Participant Score Over Study")
            plt.xticks(ticks, ticks // 24)
            plt.xlabel("Time (1 Day)")
        plt.ylabel("Score")
    plt.legend(["Sharp", "Fatigue", "Stress"])
    plt.show()

    # prints std graph for score
    for i in range(0,1) :
        stds = convert(std,i)
        plt.plot(stds, 'r')
        if type.lower() == "day" :
            plt.title("Deviation Between Individuals Over Day")
            plt.xticks([0, 3, 6, 9, 12, 15, 18, 21, 24], ['12 am', '3 am', '6 am', '9 am', '12 pm', '3 pm', '6 pm', '9 pm','12 am'])
            plt.xlabel("Time of Day")
        else :
            ticks = np.array([i for i in range(len(compressed)) if i % 24 == 0])
            plt.title("Deviation Between Individuals Over Study")
            plt.xticks(ticks, ticks // 24)
            plt.xlabel("Time (1 Day)")
        plt.ylabel("Score Standard Deviation")
    plt.legend(["Sharp", "Fatigue", "Stress"])
    plt.show()  

def analyze_ema(packed_data:tuple,type:str) :
    """
        Analyzes ema responses
    
        Parameters
        ----------
        packed_data : tuple
            6 lists created by create_data
        type: str ["Day", "Study"]
            type of data being analyzed       

    """

    # unpack data
    average, stats, std, std_stats, user_stats, user_raw_data = packed_data
    convert = lambda x,i : [t[i] for t in x]

    # plots each user line
    for line in user_raw_data :
        plt.plot(convert(line,3),alpha=0.1)
    
    # plot averaged line
    plt.plot(convert(average,3), 'r')

    if type.lower() == "day" :
        plt.title("Average Participant Response Over Day")
        plt.xticks([0, 3, 6, 9, 12, 15, 18, 21, 24], ['12 am', '3 am', '6 am', '9 am', '12 pm', '3 pm', '6 pm', '9 pm','12 am'])
        plt.xlabel("Time of Day")
    else :
        ticks = np.array([i for i in range(len(convert(average,3))) if i % 24 == 0])
        plt.title("Average Participant Response Over Study")
        plt.xticks(ticks, ticks // 24)
        plt.xlabel("Time (1 Day)")
    
    plt.ylabel("Response")
    plt.show()

    # prints stats for population, std population, and user data
    new_stats = make_features(convert(stats,3))
    new_std_stats = make_features(convert(std_stats,3))
    new_user_stats = np.average(user_stats,axis=0)

    print_data(new_stats, "Population")
    print_data(new_std_stats, "Std")
    print_data(new_user_stats, "Users")

     # prints graph for each s,s,f
    for i in range(0,3) :
        compressed = convert(average,i)

        plt.plot(compressed)
        if type.lower() == "day" :
            plt.title("Average Participant Response Over Day")
            plt.xticks([0, 3, 6, 9, 12, 15, 18, 21, 24], ['12 am', '3 am', '6 am', '9 am', '12 pm', '3 pm', '6 pm', '9 pm','12 am'])
            plt.xlabel("Time of Day")
        else :
            ticks = np.array([i for i in range(len(compressed)) if i % 24 == 0])
            plt.title("Average Participant Response Over Study")
            plt.xticks(ticks, ticks // 24)
            plt.xlabel("Time (1 Day)")
        plt.ylabel("Response")

    plt.legend(["Sharp", "Fatigue", "Stress"])
    plt.show()

    # prints std graph for each s,s,f
    for i in range(0,3) :
        stds = convert(std,i)
        plt.plot(stds)
        if type.lower() == "day" :
            plt.title("Deviation Between Individuals Over Day")
            plt.xticks([0, 3, 6, 9, 12, 15, 18, 21, 24], ['12 am', '3 am', '6 am', '9 am', '12 pm', '3 pm', '6 pm', '9 pm','12 am'])
            plt.xlabel("Time of Day")
        else :
            ticks = np.array([i for i in range(len(compressed)) if i % 24 == 0])
            plt.title("Deviation Between Individuals Over Study")
            plt.xticks(ticks, ticks // 24)
            plt.xlabel("Time (1 Day)")
        plt.xlabel("Time (1 Day)")
        plt.ylabel("Response Standard Deviation")

    plt.legend(["Sharp", "Fatigue", "Stress"])
    plt.show()  

def analyze_studies(study_data:list) :
    """
        Analyzes studies
    
        Parameters
        ----------
        study_data : list
            list of studies and data   
    """
    

    # extracts index from data
    day_index = lambda x : x[3]
    total_index = lambda x: x[3] + (x[4]*24)

    # goes through each study
    for study in study_data :
        size = get_size(study)
        # get all data daily and over study
        day_packed = create_data(study, 'day', 'ema')
        total_packed = create_data(study, 'study', 'ema')

        analyze_ema(day_packed, "Day")
        analyze_ema(total_packed, "Study")

def get_size(data) :
    """
        Gets total time in study
    
        Parameters
        ----------
        data : list
            list of data   
    """

    while isinstance(data[0], list) :
        data = flatten(data,0,2)
    size = int(np.max(data)) // 60

    return size
    
def analyze_correlations(user_data, nback_user) :
    """
        Finds correlation in the data
    
        Parameters
        ----------
        user_data : list
            list of ema data   
        nback_user : list
            list of nback data
        
    """
    
    # get all data
    nback_day_packed = create_data(nback_user, "day", "nback")
    nback_total_packed = create_data(nback_user, "study", "nback")

    day_packed_data = create_data(user_data, "day", "ema")
    total_packed_data = create_data(user_data, "study", "ema")

    nback_data = nback_day_packed[5]
    ema_data = day_packed_data[5]

    # indices for ema/nback
    i1 = 0
    i2 = 0
    i = 0

    # correlations
    coefs = []
    sf = [] # sharp fatigue
    ss = [] # sharp stress
    fs = [] # fatigue stress

    while i < min(len(nback_data),len(ema_data)) :
        # gets ids
        id1 = user_data[i1][0][0]
        id2 = nback_user[i2][0][0]

        # progress through list until ids match
        while id1 != id2 :
            while id1 < len(user_data) and id1 <= id2 :
                i1 += 1
                id1 = user_data[i1][0][0]
            while id2 < len(nback_user) and id2 <= id1 :
                i2 += 1
                id2 = user_data[i2][0][0]
        
        # gets current data 
        nback_data_cur = [d[0] for d in nback_data[i2]]
        ema_data_cur = [d[3] for d in ema_data[i1]]

        # gets all emas
        emas = [[d[j] for d in ema_data[i1]][9:21] for j in range(0,3)]

        # finds correlations
        sh_fa = np.corrcoef(emas[0], emas[1])
        sf.append(sh_fa)
        sh_st = np.corrcoef(emas[0], emas[2])
        ss.append(sh_st)
        fa_st = np.corrcoef(emas[1], emas[2])
        fs.append(fa_st)

        # gets data over day
        nback_data_cur = nback_data_cur[9:21]
        ema_data_cur = ema_data_cur[9:21]

        # gets ema vs nback coef
        coef = np.corrcoef(nback_data_cur, ema_data_cur)
        coefs.append(coef[0][1])
        i += 1
    # prints average nback/ema coef, then each ema coef
    print(np.average(coefs))

    print(np.average(sf))
    print(np.average(ss))
    print(np.average(fs))

    nback_data = nback_total_packed[1]
    ema_data = total_packed_data[1]

    # gets zeroless data
    nback_data = [d[0] for d in nback_data]
    ema_data = [d[3] for d in ema_data]

    # gets min size
    minsize = min(len(nback_data), len(ema_data))

    nback_data = nback_data[:minsize]
    ema_data = ema_data[:minsize]

    # print coef across day
    coef = np.corrcoef(nback_data, ema_data)
    print(coef)
    
def main() :

    # needs converted ema and nback file
    data = read_ema('./ema_new.csv')
    nback = read_nback('./nback.csv')

    # gets data for each study, then each user, then all responses
    study_data = flatten(data,0,1)
    user_data = flatten(data,0,2)
    all_responses = flatten(data, 0, 3)

    nback_studies = flatten(nback, 0, 1)
    nback_user = flatten(nback,0,2)
    all_nback = flatten(nback,0,3)

    # EXAMPLE

    # analyze some data - ema, day
    all_ema_day = create_data(user_data, "day", "ema")
    # returns 6 lists of data in tuple: averaged, averaged without empty spots, std between people, std without empty, user computed stats, user raw stats

    # graph the data
    analyze_ema(all_ema_day, "day")

    # analyze nback over study
    all_nback_study = create_data(nback_user, "study", "nback")
    analyze_nback(all_nback_study, "study")

    # analyze response times of day across ema, then over study
    analyze_times(all_responses, "day")
    analyze_times(all_nback, "study")
    
    # find correlations
    analyze_correlations(user_data, nback_user)

    # analyze every ema study
    analyze_studies(study_data)

if __name__ == '__main__' :
    main()