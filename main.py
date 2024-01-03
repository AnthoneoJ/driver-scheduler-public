VERSION = 8.1
"""
change , to . in distance values
"""

import os, datetime, copy, random, pathlib
from typing import Dict, List

import pandas as pd

### Duties params ###
Depot_List = [] # where drivers start & end and have meal breaks
TRAVEL_AS_PASSENGER = True
STRETCH_DUR_MIN = 7*60 + 10 # 7hr 10m
STRAIGHT_DUR_MAX = 9*60
SPLIT_DUR_MAX = 11*60 # reducing this to Straight_Dur_Max will disable split duties
SPLIT_SUM_SPELL = 9*60
SPLIT_MEAL_BREAK = 2*60
SPLIT_BREAK_START = datetime.datetime(year=2023,month=1,day=1,hour=14,minute=0)
SPLIT_BREAK_END = datetime.datetime(year=2023,month=1,day=1,hour=16,minute=0)
S_BREAK = 30
S_MIN_BREAK = 10
L_SPELL_MAX = 4*60 + 30
L_SPELL2_MAX = 3*60 + 45
L_BREAK = 45
L_BREAK1 = 15 # must be smaller than L_BREAK2
L_BREAK2 = 30

### GRASP params ###
MAX_ITER = 7
ALPHA = 0.8 # 0.0 - pure greedy selection, 1.0 - pure random selection
BROKEN_PROB = 0.1 # 0.0 - disable broken shift, 1.0 - always broken shift if possible
VNS_RADIUS = 1 # amount of trips closest in time to trip_pointer to swap with
FACTOR_DUTIES = 1000.0
FACTOR_SPELL = -1.0
FACTOR_NONDIRECT = 0.1

### Internal params ###
ENFORCE_GRASP = False
SAVE_ALL_DUTIES = True

class Duty:
    def __init__(self, trip_list: list, connecting_trip_list: list, break_list: list) -> None:
        self.trips = trip_list
        self.connections = connecting_trip_list
        self.breaks = break_list

    def sort_lists(self, reverse_order = False):
        self.trips.sort(key=self.take_ot, reverse=reverse_order)
        self.connections.sort(key=self.take_ot, reverse=reverse_order)
        self.breaks.sort(key=self.take_ot, reverse=reverse_order)

    # use ot for sorting
    def take_ot(self, elem):
        return elem[2]

class MockThread:
    def __init__(self) -> None:
        self.input_dir_path = ''
        self.trip_list = []

    def start_thread(self) -> None:
        outputs = dict()
        print("Input folder:", self.input_dir_path)
        filename_list = []
        for file in os.listdir(self.input_dir_path):
            filename = os.fsdecode(file)
            filename_list.append(filename)
        filename_list.sort(key=self.take_suffix)
        for filename in filename_list:
            if not (filename.endswith('.xlsx') or filename.endswith('.xls')):
                continue
            print("Processing", filename)
            input_df = pd.read_excel(os.path.join(self.input_dir_path, filename))
            input_labels = input_df.columns
            # 0: CODE TRIP; 1: LINE; 2: ORIGIN TIME; 3: ORIGIN POINT; 4: DESTINATION TIME;
            # 5: DESTINATION POINT; 6: DISTANCE; 7: BUS
            input_df.sort_values(by=[input_labels[2]], inplace=True)
            self.trip_list = input_df.values.tolist()
            for i in range(len(self.trip_list)):
                dist_str = str(self.trip_list[i][6])
                if dist_str.count(',')==1:
                    dist_str = dist_str.replace(',','.') # eg 1,59 -> 1.59
                else:
                    dist_str = dist_str.replace(',','') # eg 6,346,735 -> 6346735
                self.trip_list[i][6] = float(dist_str)
            ot_example = self.trip_list[0][2]
            if isinstance(ot_example, datetime.time):
                self.trip_list = self.attach_date(self.trip_list)
            if len(Depot_List)==0:
                self.update_depot()

            # use logic based algorithm to create schedule
            self.all_duties = self.logic_algorithm()
            logic_solution = self.duties_to_chromosome(self.all_duties)
            # assign connecting trips, assign breaks, append output
            logic_schedule = self.generate_schedule(logic_solution)
            outputs[filename] = logic_schedule
            logic_cost = self.calculate_cost(logic_schedule)

            if not (ENFORCE_GRASP or len(self.all_duties)==0):
                if SAVE_ALL_DUTIES:
                    self.duties_to_csv()
                continue

            print('Running GRASP')
            grasp_schedule = self.run_grasp()
            grasp_cost = self.calculate_cost(grasp_schedule)
            """ if logic_cost < grasp_cost:
                grasp_schedule = logic_schedule """
            # assign connecting trips, assign breaks, append output
            outputs[filename] = grasp_schedule

        self.output_to_xlsx(outputs)

    ######################
    ### MAIN FUNCTIONS ###
    ######################
    def logic_algorithm(self):
        """
        TODO Note: Max_Bus_Per_Duty and Max_Trips_Per_Duty filters are not used in this algorithm
        All breaks are assigned according to long line rules, which also meet short line's rules for dur > 7.5hr
        """
        global Depot_List
        
        remaining_trip_i_list = [i for i in range(len(self.trip_list))]
        duties_sol = []
        while(len(remaining_trip_i_list)>0):
            first_trip_i = remaining_trip_i_list[0]
            # if first trip not from depot
            if not self.trip_list[remaining_trip_i_list[0]][3] in Depot_List:
                if TRAVEL_AS_PASSENGER and remaining_trip_i_list[0]>0:
                    # get latest connecting trip from depot
                    connect_trip_exist = False
                    for i in range(remaining_trip_i_list[0]-1,-1,-1):
                        trip = self.trip_list[i]
                        trip_from_depot = trip[3] in Depot_List
                        trip_arrive_here = trip[5] == self.trip_list[remaining_trip_i_list[0]][3]
                        trip_arrive_before = trip[4] <= self.trip_list[remaining_trip_i_list[0]][2]
                        if trip_from_depot and trip_arrive_here and trip_arrive_before:
                            first_trip_i = i
                            connect_trip_exist = True
                            break
                    if connect_trip_exist == False:
                        duties_sol = []
                        return duties_sol # infeasible, redirect to grasp
                else:
                    duties_sol = []
                    return duties_sol # infeasible, redirect to grasp
                
            new_duty = [remaining_trip_i_list[0]]
            remaining_trip_i_list.pop(0) # if this is last item, loop will exit
            index_to_remove = []
            for i in range(len(remaining_trip_i_list)):
                is_valid = self.is_valid_next_trip(i, remaining_trip_i_list, new_duty, first_trip_i)
                if is_valid:
                    new_duty.append(remaining_trip_i_list[i])
                    index_to_remove.append(i)
            # check if last trip arrive depot
            if not self.trip_list[new_duty[-1]][5] in Depot_List:
                if TRAVEL_AS_PASSENGER:
                    # check for any possible connecting trip to depot
                    connect_trip_exist = False
                    for trip in self.trip_list:
                        trip_to_depot = trip[5] in Depot_List
                        trip_from_here = trip[3] == self.trip_list[new_duty[-1]][5]
                        trip_depart_after = trip[2] >= self.trip_list[new_duty[-1]][4]
                        if trip_to_depot and trip_from_here and trip_depart_after:
                            connect_trip_exist = True
                            break
                    if connect_trip_exist == False:
                        duties_sol = []
                        return duties_sol # infeasible, redirect to grasp
            duties_sol.append(new_duty)

            ori_len = len(remaining_trip_i_list)
            for delete_i in range(ori_len-1,-1,-1):
                if delete_i in index_to_remove:
                    remaining_trip_i_list.pop(delete_i)

        return duties_sol
    
    def run_grasp(self) -> Dict[int,Duty]:
        """Only self.trip_list remain as original, all other lists contain indexes to self.trip_list"""
        best_schedule = None
        best_cost = float('inf')
        print('Iterations:')
        for iter_num in range(MAX_ITER):
            print('\r',iter_num+1,end='')
            # Construction phase
            candidate_solution = []
            remaining_trips = [trip_i for trip_i in range(len(self.trip_list))]
            remaining_trips.sort(key=self.take_ot)
            max_count = len(remaining_trips) # to exit loop as some trips might not have any suitable pairing
            count = 0
            while len(remaining_trips)>0 and count<=max_count:
                count += 1
                if len(remaining_trips) == 1: # i.e. last remaining trip
                    new_duty = [remaining_trips[0]]
                    candidate_solution.append(new_duty)
                    remaining_trips.pop(0)
                    break

                trips_from_depot = [] # tuples of (trip_i,connecting_trip_i)
                for trip_i in remaining_trips:
                    first_trip_i = trip_i
                    # if first trip not from depot
                    if self.trip_list[trip_i][3] in Depot_List:
                        trips_from_depot.append((trip_i,first_trip_i))
                    else:
                        if TRAVEL_AS_PASSENGER and trip_i>0:
                            # get latest connecting trip from depot
                            for i in range(trip_i-1,-1,-1):
                                trip = self.trip_list[i]
                                trip_from_depot = trip[3] in Depot_List
                                trip_arrive_here = trip[5] == self.trip_list[trip_i][3]
                                trip_arrive_before = trip[4] <= self.trip_list[trip_i][2]
                                if trip_from_depot and trip_arrive_here and trip_arrive_before:
                                    first_trip_i = i
                                    trips_from_depot.append((trip_i, first_trip_i))
                                    break
                if len(trips_from_depot) == 0:
                    break
                
                j, first_trip_i = trips_from_depot[0]# random.choice(trips_from_depot)
                new_duty = []
                new_duty.append(j)
                remaining_trips.remove(j)

                while True:
                    rcl = self.find_candidate_trips(new_duty, remaining_trips, first_trip_i)
                    if len(rcl) == 0:
                        candidate_solution.append(new_duty)
                        break
                    else:
                        j = random.choice(rcl)
                        new_duty.append(j)
                        remaining_trips.remove(j)
            if len(remaining_trips) > 0:
                for trip_i in remaining_trips:
                    candidate_solution.append([trip_i]) # new duty per each unpaired trip

            # Local Search - order by ot, merge, is_valid_duty 
            # reset local everytime a merge occur
            # & VNS(test if changing each trip to another trip close in time improves)
            pointer = -1
            while True:
                pointer += 1
                if pointer >= len(candidate_solution)-1:
                    break
                for pointer_next in range(pointer+1,len(candidate_solution)):
                    merged_duty = []
                    merged_duty.extend(candidate_solution[pointer])
                    merged_duty.extend(candidate_solution[pointer_next])
                    merged_duty.sort(key=self.take_ot)
                    if self.is_valid_duty(merged_duty):
                        candidate_solution[pointer] = merged_duty
                        candidate_solution.pop(pointer_next)
                        pointer = -1
                        break
            # VNS
            improving_pairs = [] # list of (trip_i,pointed_trip_i,swapped_cost)
            candidate_chrom = self.duties_to_chromosome(candidate_solution)
            candidate_schedule = self.generate_schedule(candidate_chrom)
            candidate_cost = self.calculate_cost(candidate_schedule)
            for trip_i,duty_name in enumerate(candidate_chrom):
                # only need to search downwards since upwards search is mutually covered
                # e.g. swapping index pair (1,3) is the same as swapping (3,1)
                for pointed_trip_i in range(trip_i+1,min(trip_i+VNS_RADIUS+1,len(candidate_chrom))):
                    reference_duty = copy.deepcopy(candidate_solution[duty_name])
                    reference_duty[reference_duty.index(trip_i)] = pointed_trip_i
                    pointed_duty = copy.deepcopy(candidate_solution[candidate_chrom[pointed_trip_i]])
                    pointed_duty[pointed_duty.index(pointed_trip_i)] = trip_i
                    if self.is_valid_duty(reference_duty) and self.is_valid_duty(pointed_duty):
                        swapped_duty_list = copy.deepcopy(candidate_solution)
                        swapped_duty_list[duty_name] = reference_duty
                        swapped_duty_list[candidate_chrom[pointed_trip_i]] = pointed_duty
                        swapped_chrom = self.duties_to_chromosome(swapped_duty_list)
                        swapped_schedule = self.generate_schedule(swapped_chrom)
                        swapped_cost = self.calculate_cost(swapped_schedule)
                        if swapped_cost < candidate_cost:
                            improving_pairs.append((trip_i,pointed_trip_i,swapped_cost))
            improving_pairs.sort(key=self.take_swapped_cost)
            while(len(improving_pairs) > 0):
                chosen_trip_i, chosen_pointed_trip_i, _ = improving_pairs[0]
                ref_copy = copy.deepcopy(candidate_chrom[chosen_trip_i])
                candidate_chrom[chosen_trip_i] = copy.deepcopy(candidate_chrom[chosen_pointed_trip_i])
                candidate_chrom[chosen_pointed_trip_i] = ref_copy
                improving_pairs.pop(0)
                for i in range(len(improving_pairs)-1,-1,-1):
                    # remove duties at which trips have been swapped. further mingling these duties could
                    # potentially violate the duties' validity
                    chosen_pair_duties = [candidate_chrom[chosen_trip_i], candidate_chrom[chosen_pointed_trip_i]]
                    trip_i, pointed_trip_i, _ = improving_pairs[i]
                    if ((candidate_chrom[trip_i] in chosen_pair_duties) or
                        (candidate_chrom[pointed_trip_i] in chosen_pair_duties)):
                        improving_pairs.pop(i)

            candidate_schedule = self.generate_schedule(candidate_chrom)
            candidate_cost = self.calculate_cost(candidate_schedule)
            if candidate_cost < best_cost:
                best_schedule = candidate_schedule
                best_cost = candidate_cost
        print('')
        return best_schedule

    def find_candidate_trips(self,existing_trips:list[int],remaining_trips:list[int],first_trip_i:int) -> list[int]:
        restricted_candidate_list = []
        valid_trips = []
        for pointer,trip_i in enumerate(remaining_trips):
            if self.is_valid_next_trip(pointer,remaining_trips,existing_trips,first_trip_i,BROKEN_PROB):
                valid_trips.append(trip_i)
        if len(valid_trips) > 0:
            valid_trips.sort(key=self.take_ot)
            tdiff_min = self.minute_diff(self.trip_list[existing_trips[-1]][4], self.trip_list[valid_trips[0]][2])
            tdiff_max = self.minute_diff(self.trip_list[existing_trips[-1]][4], self.trip_list[valid_trips[-1]][2])
            tdiff_limit = tdiff_min + ALPHA*(tdiff_max - tdiff_min)
            for trip_i in valid_trips:
                tdiff = self.minute_diff(self.trip_list[existing_trips[-1]][4], self.trip_list[trip_i][2])
                if tdiff <= tdiff_limit:
                    restricted_candidate_list.append(trip_i)

        return restricted_candidate_list

    def is_valid_duty(self, duty: list[int]) -> bool:
        """
        candidate_duty: a combination of indexes to self.trip_list e.g. [3,5,6,24,59]
        """
        global Depot_List

        # check whether start & end at depot
        if not self.trip_list[duty[0]][3] in Depot_List:
            if TRAVEL_AS_PASSENGER:
                # check for any possible connecting trip from depot
                connect_trip_exist = False
                for trip in self.trip_list:
                    trip_from_depot = trip[3] in Depot_List
                    trip_arrive_here = trip[5] == self.trip_list[duty[0]][3]
                    trip_arrive_before = trip[4] <= self.trip_list[duty[0]][2]
                    if trip_from_depot and trip_arrive_here and trip_arrive_before:
                        connect_trip_exist = True
                        break
                if connect_trip_exist == False:
                    return False
            else:
                return False
        if not self.trip_list[duty[-1]][5] in Depot_List:
            if TRAVEL_AS_PASSENGER:
                # check for any possible connecting trip to depot
                connect_trip_exist = False
                for trip in self.trip_list:
                    trip_to_depot = trip[5] in Depot_List
                    trip_from_here = trip[3] == self.trip_list[duty[-1]][5]
                    trip_depart_after = trip[2] >= self.trip_list[duty[-1]][4]
                    if trip_to_depot and trip_from_here and trip_depart_after:
                        connect_trip_exist = True
                        break
                if connect_trip_exist == False:
                    return False
            else:
                return False

        # check that trips are ordered sequentially and for any overlap
        for i in range(len(duty)-1):
            # ...sequntial
            if not (self.trip_list[duty[i]][4] <= self.trip_list[duty[i+1]][2]):
                return False
            # ...overlap
            for j in range(i+1,len(duty),1):
                # if not trip_i before trip_j or trip_i after trip_j
                if not (self.trip_list[duty[i]][4]<=self.trip_list[duty[j]][2] or
                        self.trip_list[duty[i]][2]>=self.trip_list[duty[j]][4]):
                    return False

        # check if there's a connectring trip for non-direct sequence of trips
        for i in range(1,len(duty)):
            if self.trip_list[duty[i-1]][5] != self.trip_list[duty[i]][3]:
                if TRAVEL_AS_PASSENGER:
                    # check if possible to travel as passenger
                    connect_trip_exist = False
                    for trip in self.trip_list:
                        is_connecting_trip = (trip[3]==self.trip_list[duty[i-1]][5] and
                                              trip[5]==self.trip_list[duty[i]][3])
                        is_between = trip[2]>=self.trip_list[duty[i-1]][4] and trip[4]<=self.trip_list[duty[i]][2]
                        if is_connecting_trip and is_between:
                            connect_trip_exist = True
                            break
                    if connect_trip_exist == False:
                        return False
                else:
                    return False

        stretch_dur, first_trip_i, last_trip_i = self.get_stretch_info(duty)

        # check if duty is too short or too long
        if stretch_dur < STRETCH_DUR_MIN or stretch_dur > SPLIT_DUR_MAX:
            return False

        duty_distance = self.get_duty_distance(duty)

        if stretch_dur <= STRAIGHT_DUR_MAX and duty_distance < 50: # straight & short duty
            # check if there are enough gaps to meet min break duration
            total_break = self.get_break_dur_s(duty)
            if total_break < S_BREAK:
                return False
        elif stretch_dur <= STRAIGHT_DUR_MAX and duty_distance >= 50: # straight & long duty
            # check if break exist after driving L_Spell_Max mins
            driving_dur = self.minute_diff(self.trip_list[duty[0]][2],self.trip_list[duty[0]][4])
            break_dur = 0
            for i in range(len(duty)-1):
                gap_dur = self.minute_diff(self.trip_list[duty[i]][4],self.trip_list[duty[i+1]][2])
                if (gap_dur>=L_BREAK) or (break_dur>=L_BREAK1 and gap_dur>=L_BREAK2):
                    driving_dur = self.minute_diff(self.trip_list[duty[i+1]][2],self.trip_list[duty[i+1]][4])
                    break_dur = 0
                elif break_dur==0 and gap_dur>=L_BREAK1:
                    driving_dur += self.minute_diff(self.trip_list[duty[i]][4],self.trip_list[duty[i+1]][4])
                    break_dur += gap_dur
                else:
                    driving_dur += self.minute_diff(self.trip_list[duty[i]][4],self.trip_list[duty[i+1]][4])
                if driving_dur > L_SPELL_MAX:
                    return False
        elif stretch_dur > STRAIGHT_DUR_MAX: # split duty (short/long)
            # create/don't create split duty based on probability
            random_val = random.uniform(0.0, 1.0)
            if random_val >= BROKEN_PROB:
                return False
            # check if total spell duration is less than Split_Sum_Spell
            is_short = duty_distance < 50
            total_break_dur = self.get_break_dur(duty, is_short, first_trip_i, last_trip_i)
            total_spell_dur = stretch_dur - total_break_dur
            if total_spell_dur > SPLIT_SUM_SPELL:
                return False
            # check if there's enough time for meal break in middle gap
            meal_break_possible = False
            for i in range(1,len(duty),1):
                if self.trip_list[duty[i-1]][4] < (self.trip_list[first_trip_i][2] +
                                                   datetime.timedelta(minutes=L_SPELL2_MAX)):
                    continue
                if self.trip_list[duty[i]][2] > (self.trip_list[last_trip_i][4] -
                                                 datetime.timedelta(minutes=L_SPELL2_MAX)):
                    continue
                if (not self.trip_list[duty[i-1]][5] in Depot_List and
                    not self.trip_list[duty[i]][3] in Depot_List):
                    continue
                meal_break_start = self.trip_list[duty[i-1]][4]
                meal_break_end = self.trip_list[duty[i]][2]
                gap_dur = self.minute_diff(meal_break_start,meal_break_end)
                if (self.trip_list[duty[i-1]][5]!=self.trip_list[duty[i]][3] and
                    self.trip_list[duty[i-1]][5] in Depot_List):
                    # find last connecting trip to duty[i][3]
                    _, lct_i = self.get_connect_trip_i_el(duty[i-1],duty[i])
                    meal_break_end = self.trip_list[lct_i][2]
                    offset_right = self.minute_diff(meal_break_end,self.trip_list[duty[i]][2])
                    gap_dur -= offset_right
                elif (self.trip_list[duty[i-1]][5] != self.trip_list[duty[i]][3] and
                      self.trip_list[duty[i]][3] in Depot_List):
                    # find earliest connecting trip to duty[i][3]
                    ect_i, _ = self.get_connect_trip_i_el(duty[i-1],duty[i])
                    meal_break_start = self.trip_list[ect_i][4]
                    offset_left = self.minute_diff(self.trip_list[duty[i-1]][4],meal_break_start)
                    gap_dur -= offset_left
                break_lowbound = datetime.datetime.combine(meal_break_start.date(),SPLIT_BREAK_START.time())
                break_uppbound = break_lowbound + (SPLIT_BREAK_END - SPLIT_BREAK_START)
                is_in_bound = meal_break_start>=break_lowbound and meal_break_end<=break_uppbound
                if gap_dur < SPLIT_MEAL_BREAK or (not is_in_bound):
                    continue
                else:
                    meal_break_possible = True
                    break # meal break possible
            if meal_break_possible == False:
                return False

        return True

    def is_valid_next_trip(self, pointer, remaining_trip_i_list, new_duty, first_trip_i, split_prob=0.0) -> bool:
        """
        pointer is an index to remaining_trip_i_list i.e. the trip to be added
        first_trip_i is first trip in new_duty if it departs from depot, otherwise it is trip_i of a
        connecting trip
        """
        global Depot_List

        pointer_i = remaining_trip_i_list[pointer]
        pointer_t = self.trip_list[pointer_i]
        last_t = self.trip_list[new_duty[-1]]

        # check trip is after last trip in new_duty
        if not (last_t[4] <= pointer_t[2]):
            return False
        
        # check if stretch exceed SPLIT_DUR_MAX/STRAIGHT_DUR_MAX
        random_val = random.uniform(0.0, 1.0)
        if random_val < split_prob:
            # will produce split duty type
            if self.minute_diff(self.trip_list[first_trip_i][2],pointer_t[4]) > SPLIT_DUR_MAX:
                return False
        else:
            # will produce straight duty type
            if self.minute_diff(self.trip_list[first_trip_i][2],pointer_t[4]) > STRAIGHT_DUR_MAX:
                return False
        
        # check if connecting trip exist for non-direct trip
        if last_t[5] != pointer_t[3]:
            if TRAVEL_AS_PASSENGER:
                # check if possible to travel as passenger
                connect_trip_exist = False
                for trip in self.trip_list:
                    is_connecting_trip = (trip[3]==last_t[5] and trip[5]==pointer_t[3])
                    is_between = trip[2]>=last_t[4] and trip[4]<=pointer_t[2]
                    if is_connecting_trip and is_between:
                        connect_trip_exist = True
                        break
                if connect_trip_exist == False:
                    return False
            else:
                return False
            
        temp_duty = []
        temp_duty.extend(new_duty)
        temp_duty.append(pointer_i)

        # check whether spell length exceed L_Spell_Max if pointer_t is added
        first_i = 0 # initiate to first trip in new_duty
        # get first trip after last break
        for i in range(len(temp_duty)-1,0,-1):
            trip_before = self.trip_list[temp_duty[i-1]]
            trip_after = self.trip_list[temp_duty[i]]
            if self.minute_diff(trip_before[4],trip_after[2]) >= L_BREAK:
                first_i = i
        if self.minute_diff(self.trip_list[temp_duty[first_i]][2],pointer_t[4]) > L_SPELL_MAX:
            return False
        
        # if STRETCH_DUR_MIN exceeded & pointer_t does not end at depot, find earliest connecting trip
        # then check if spell length exceed L_Spell_Max
        min_dur_met = self.minute_diff(self.trip_list[first_trip_i][2],pointer_t[4]) >= STRETCH_DUR_MIN
        if TRAVEL_AS_PASSENGER and min_dur_met and (not self.trip_list[temp_duty[-1]][5] in Depot_List):
            last_trip_i = None
            trip_dt_earliest = self.trip_list[-1][4] # assuming self.trip_list is ordered
            for trip_i in range(len(self.trip_list)):
                trip_to_depot = self.trip_list[trip_i][5] in Depot_List
                trip_from_here = self.trip_list[trip_i][3] == self.trip_list[temp_duty[-1]][5]
                trip_depart_after = self.trip_list[trip_i][2] >= self.trip_list[temp_duty[-1]][4]
                if trip_to_depot and trip_from_here and trip_depart_after:
                    trip_dt = self.trip_list[trip_i][4]
                    if trip_dt <= trip_dt_earliest:
                        trip_dt_earliest = trip_dt
                        last_trip_i = trip_i
            if last_trip_i == None:
                # no connecting trip to depot found
                return False
            else:
                # check stretch duration
                duty_dur = self.minute_diff(self.trip_list[first_trip_i][2], self.trip_list[last_trip_i][4])
                if duty_dur > STRAIGHT_DUR_MAX:
                    return False
            
        return True
    
    def calculate_cost(self, duties: Dict[int,Duty]) -> float:
        global Depot_List
        # 1. Number of duties
        num_of_duties = len(duties)

        # 2. Total spell lengths (i.e. not including meal breaks)
        total_spell_length = 0
        for duty_name in duties:
            d = duties[duty_name]
            d.sort_lists()
            start_time = d.trips[0][2]
            end_time = d.trips[-1][4]
            if len(d.connections) > 0:
                if d.connections[0][2] < d.trips[0][2]:
                    start_time = d.connections[0][2]
                if d.connections[-1][4] > d.trips[-1][2]:
                    end_time = d.connections[-1][4]
            d_total_break = 0
            for b in d.breaks:
                d_total_break += self.minute_diff(b[2],b[4])
            d_spell_len = self.minute_diff(start_time,end_time) - d_total_break
            total_spell_length += d_spell_len

        # 3. Number of non-direct trips
        non_direct_count = 0
        for duty_name in duties:
            non_direct_count += len(duties[duty_name].connections)
        """ for d in duties:
            for i,t in enumerate(d):
                if (i == 0) and not self.trip_list[t][3] in Depot_List:
                    non_direct_count += 1
                if (i == len(d)-1) and not self.trip_list[t][5] in Depot_List:
                    non_direct_count += 1
                if (i < len(d)-1) and (self.trip_list[d[i]][5] != self.trip_list[d[i+1]][3]):
                    non_direct_count += 1 """
        
        cost = FACTOR_DUTIES*num_of_duties + FACTOR_SPELL*total_spell_length + FACTOR_NONDIRECT*non_direct_count
        return cost

    def update_depot(self):
        global Depot_List
        for trip in self.trip_list:
            if not trip[3] in Depot_List:
                Depot_List.append(trip[3])
            if not trip[5] in Depot_List:
                Depot_List.append(trip[5])

    def generate_schedule(self, trip_i_list: list) -> Dict[int,Duty]:
        global Depot_List

        duties: Dict[int,Duty] = dict()
        none_count = 0
        for trip_i in range(len(trip_i_list)):
            if trip_i_list[trip_i]==None:
                # create 1 duty per None trip
                none_count -= 1
                duties[none_count] = Duty([self.trip_list[trip_i]],[],[])
            elif not trip_i_list[trip_i] in duties:
                duties[trip_i_list[trip_i]] = Duty([self.trip_list[trip_i]],[],[])
            else:
                duties[trip_i_list[trip_i]].trips.append(self.trip_list[trip_i])

        for duty_i in duties:
            duties[duty_i].sort_lists()
            duty_start = duties[duty_i].trips[0][2]
            duty_end = duties[duty_i].trips[-1][4]

            # if not starting from depot
            if not duties[duty_i].trips[0][3] in Depot_List:
                if not TRAVEL_AS_PASSENGER:
                    print('Infeasible sol. Duty not starting in depot but TRAVEL_AS_PASSENGER is false!')
                assert TRAVEL_AS_PASSENGER==True
                possible_connections = []
                for trip in self.trip_list:
                    trip_from_depot = trip[3] in Depot_List
                    trip_arrive_here = trip[5] == duties[duty_i].trips[0][3]
                    trip_arrive_before = trip[4] <= duties[duty_i].trips[0][2]
                    if trip_from_depot and trip_arrive_here and trip_arrive_before:
                        possible_connections.append(trip)
                possible_connections.sort(key=self.take_ot, reverse=True)
                # pick the latest starting connection (to reduce overall duty duration)
                duties[duty_i].connections.append(possible_connections[0])
                duty_start = possible_connections[0][2]

            # if not ending at depot
            if not duties[duty_i].trips[-1][5] in Depot_List:
                if not TRAVEL_AS_PASSENGER:
                    print('Infeasible sol. Duty not ending in depot but TRAVEL_AS_PASSENGER is false!')
                assert TRAVEL_AS_PASSENGER==True
                possible_connections = []
                for trip in self.trip_list:
                    trip_to_depot = trip[5] in Depot_List
                    trip_from_here = trip[3] == duties[duty_i].trips[-1][5]
                    trip_depart_after = trip[2] >= duties[duty_i].trips[-1][4]
                    if trip_to_depot and trip_from_here and trip_depart_after:
                        possible_connections.append(trip)
                possible_connections.sort(key=self.take_dt)
                # pick the earliest connection to arrive to depot
                duties[duty_i].connections.append(possible_connections[0])
                duty_end = possible_connections[0][4]

            # fill in gaps with connecting trips
            duties[duty_i].connections.extend(self.get_gap_connections(duties[duty_i].trips))

            # assign breaks, update connecting trips if necessary
            duty_dur = self.minute_diff(duty_start,duty_end)
            duty_distance = self.get_duty_distance(self.triplist_to_duty(duties[duty_i].trips))
            if duty_dur <= STRAIGHT_DUR_MAX and duty_distance < 50: # straight & short duty
                for i in range(len(duties[duty_i].trips)-1):
                    gap_dur = self.minute_diff(duties[duty_i].trips[i][4],duties[duty_i].trips[i+1][2])
                    if gap_dur >= S_MIN_BREAK:
                        gap_break = self.build_break(duties[duty_i].trips[i],duties[duty_i].trips[i+1])
                        duties[duty_i].breaks.append(gap_break)
            elif duty_dur <= STRAIGHT_DUR_MAX and duty_distance >= 50: # straight & long duty
                driving_dur = self.minute_diff(duties[duty_i].trips[0][2],duties[duty_i].trips[0][4])
                break_dur = 0
                L_Break1_checkpoint = None
                for i in range(len(duties[duty_i].trips)-1):
                    gap_dur = self.minute_diff(duties[duty_i].trips[i][4],duties[duty_i].trips[i+1][2])
                    if gap_dur >= L_BREAK:
                        gap_break = self.build_break(duties[duty_i].trips[i],duties[duty_i].trips[i+1])
                        duties[duty_i].breaks.append(gap_break)
                        driving_dur = self.minute_diff(duties[duty_i].trips[i+1][2],duties[duty_i].trips[i+1][4])
                        break_dur = 0
                        L_Break1_checkpoint = None
                    elif break_dur>=L_BREAK1 and gap_dur>=L_BREAK2:
                        gap_break_lb1 = self.build_break(duties[duty_i].trips[L_Break1_checkpoint],
                                                         duties[duty_i].trips[L_Break1_checkpoint+1])
                        duties[duty_i].breaks.append(gap_break_lb1)
                        gap_break_lb2 = self.build_break(duties[duty_i].trips[i],duties[duty_i].trips[i+1])
                        duties[duty_i].breaks.append(gap_break_lb2)
                        driving_dur = self.minute_diff(duties[duty_i].trips[i+1][2],duties[duty_i].trips[i+1][4])
                        break_dur = 0
                        L_Break1_checkpoint = None
                    elif break_dur==0 and gap_dur>=L_BREAK1:
                        driving_dur += self.minute_diff(duties[duty_i].trips[i][4],duties[duty_i].trips[i+1][4])
                        break_dur += gap_dur
                        L_Break1_checkpoint = i
                    else:
                        driving_dur += self.minute_diff(duties[duty_i].trips[i][4],duties[duty_i].trips[i+1][4])
            elif duty_dur > STRAIGHT_DUR_MAX: # split duty (short/long)
                # check if there's enough time for meal break in middle gap
                trip_list = duties[duty_i].trips
                for i in range(len(trip_list)-1):
                    if trip_list[i][4] < duty_start + datetime.timedelta(minutes=L_SPELL2_MAX):
                        continue
                    if trip_list[i+1][2] > duty_end - datetime.timedelta(minutes=L_SPELL2_MAX):
                        continue
                    if not trip_list[i][5] in Depot_List and not trip_list[i+1][3] in Depot_List:
                        continue
                    gap_dur = self.minute_diff(trip_list[i][4],trip_list[i+1][2])
                    lct = None
                    ect = None
                    if trip_list[i][5] != trip_list[i+1][3] and trip_list[i][5] in Depot_List:
                        # find last connecting trip to duty[i+1][3]
                        _, lct_i = self.get_connect_trip_i_el(self.trip_to_tripi(trip_list[i]),
                                                            self.trip_to_tripi(trip_list[i+1]))
                        lct = self.trip_list[lct_i]
                        offset_right = self.minute_diff(lct[2],trip_list[i+1][2])
                        gap_dur -= offset_right
                    elif trip_list[i][5] != trip_list[i+1][3] and trip_list[i+1][3] in Depot_List:
                        # find earliest connecting trip to duty[i+1][3]
                        ect_i, _ = self.get_connect_trip_i_el(self.trip_to_tripi(trip_list[i]),
                                                              self.trip_to_tripi(trip_list[i+1]))
                        ect = self.trip_list[ect_i]
                        offset_left = self.minute_diff(trip_list[i][4],ect[4])
                        gap_dur -= offset_left
                    if gap_dur < SPLIT_MEAL_BREAK:
                        continue
                    elif lct!=None or ect!=None:
                        # remove connecting trips in meal break gap first
                        filtered_connections = []
                        for ct in duties[duty_i].connections:
                            is_connecting_trip = ct[3]==trip_list[i][5] and ct[5]==trip_list[i+1][3]
                            is_between = ct[2]>=trip_list[i][4] and ct[4]<=trip_list[i+1][2]
                            if not (is_connecting_trip and is_between):
                                filtered_connections.append(ct)
                        duties[duty_i].connections = filtered_connections

                        if lct!=None:
                            duties[duty_i].connections.append(lct)
                            gap_break = self.build_break(trip_list[i],lct)
                            duties[duty_i].breaks.append(gap_break)
                        else:
                            duties[duty_i].connections.append(ect)
                            gap_break = self.build_break(ect,trip_list[i+1])
                            duties[duty_i].breaks.append(gap_break)
                        break
                    else:
                        gap_break = self.build_break(trip_list[i],trip_list[i+1])
                        duties[duty_i].breaks.append(gap_break)
                        break
                duties[duty_i].sort_lists()
                
        return duties
    
    def get_duty_distance(self, duty: list) -> float:
        duty_distance = 0
        for trip_i in duty:
            duty_distance += float(self.trip_list[trip_i][6])

        return duty_distance
    
    def get_connect_trip_i_el(self, trip_i_before: int, trip_i_after: int) -> tuple[int,int]:
        earliest_connect_trip_i = False # index to a trip in self.trip_list
        latest_connect_trip_i = False # index to a trip in self.trip_list
        ot_pointer = self.trip_list[trip_i_after][2]
        dt_pointer = self.trip_list[trip_i_before][4]
        for trip_i in range(len(self.trip_list)):
            is_connecting_loc = (self.trip_list[trip_i][3]==self.trip_list[trip_i_before][5] and
                                 self.trip_list[trip_i][5]==self.trip_list[trip_i_after][3])
            is_between_time = (self.trip_list[trip_i][2]>=self.trip_list[trip_i_before][4] and
                               self.trip_list[trip_i][4]<=self.trip_list[trip_i_after][2])
            if is_connecting_loc and is_between_time:
                if self.trip_list[trip_i][2]<=ot_pointer:
                    earliest_connect_trip_i = trip_i
                    ot_pointer = self.trip_list[trip_i][2]
                if self.trip_list[trip_i][4]>=dt_pointer:
                    latest_connect_trip_i = trip_i
                    dt_pointer = self.trip_list[trip_i][4]

        return earliest_connect_trip_i, latest_connect_trip_i

    def output_to_xlsx(self, outputs_dict: Dict[str,Dict[int,Duty]]) -> None:
        output_dir = os.path.join(str(pathlib.Path.home()), 'Desktop', 'bdsp_output')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        duties_dir = os.path.join(output_dir, 'duties') # input duties for rostering
        if not os.path.exists(duties_dir):
            os.mkdir(duties_dir)
        print('Output directory:', output_dir)
        header = ['CODE TRIP', 'LINE', 'ORIGIN TIME', 'ORIGIN POINT', 'DESTINATION TIME',
                'DESTINATION POINT', 'DISTANCE', 'BUS', 'DUTY']
        output_df_tuples = [('', pd.DataFrame())]
        del output_df_tuples[0] # clear after declaring
        for input_fname in outputs_dict:
            duties_dict = outputs_dict[input_fname]
            df = pd.DataFrame(columns=header)
            for duty_i in duties_dict:
                duty = duties_dict[duty_i]
                for t in duty.trips:
                    row_data = [[t[0],t[1],t[2],t[3],t[4],t[5],t[6],t[7],duty_i]]
                    row_df = pd.DataFrame(row_data, columns=header)
                    df = pd.concat([df, row_df], ignore_index=True, sort=False)
                for b in duty.breaks:
                    row_data = [[b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7],duty_i]]
                    row_df = pd.DataFrame(row_data, columns=header)
                    df = pd.concat([df, row_df], ignore_index=True, sort=False)
                for c in duty.connections:
                    code_trip = 'CONNECTION_' + str(c[0])
                    row_data = [[code_trip,c[1],c[2],c[3],c[4],c[5],c[6],c[7],duty_i]]
                    row_df = pd.DataFrame(row_data, columns=header)
                    df = pd.concat([df, row_df], ignore_index=True, sort=False)
            output_df_tuples.append((input_fname, df))
        output_xlsx = os.path.join(output_dir, 'output.xlsx')
        with pd.ExcelWriter(output_xlsx) as writer:
            duties_suffix = 0 # represent day 1,2,3,...
            for input_fname, output_df in output_df_tuples:
                output_df.to_excel(writer, sheet_name=input_fname, index=False)
                duties_suffix += 1
                duties_fname = 'duties_' + str(duties_suffix) + '.csv'
                duties_csv = os.path.join(duties_dir, duties_fname)
                output_df.to_csv(duties_csv, index=False)
                print(input_fname + ' converted to ' + duties_fname)
        print('Output generated at',output_xlsx)

    #####################################
    ### is_valid_duty() sub-functions ###
    #####################################
    def get_stretch_info(self, duty: list[int]) -> tuple[int,int,int]:
        global Depot_List

        first_trip_i = duty[0] # an index to a trip in self.trip_list
        last_trip_i = duty[-1] # an index to a trip in self.trip_list
        earliest_ot = self.trip_list[duty[0]][2]
        latest_dt = self.trip_list[duty[-1]][4]
        
        trip_dur_first = 0
        if TRAVEL_AS_PASSENGER and (not self.trip_list[duty[0]][3] in Depot_List):
            # find longest connecting trip from depot
            for trip_i in range(len(self.trip_list)):
                trip_from_depot = self.trip_list[trip_i][3] in Depot_List
                trip_arrive_here = self.trip_list[trip_i][5] == self.trip_list[duty[0]][3]
                trip_arrive_before = self.trip_list[trip_i][4] <= self.trip_list[duty[0]][2]
                if trip_from_depot and trip_arrive_here and trip_arrive_before:
                    trip_dur = self.minute_diff(self.trip_list[trip_i][2],self.trip_list[trip_i][4])
                    if trip_dur > trip_dur_first:
                        trip_dur_first = trip_dur
                        first_trip_i = trip_i
        trip_dur_last = 0
        if TRAVEL_AS_PASSENGER and (not self.trip_list[duty[-1]][5] in Depot_List):
            # find longest connecting trip to depot
            for trip_i in range(len(self.trip_list)):
                trip_to_depot = self.trip_list[trip_i][5] in Depot_List
                trip_from_here = self.trip_list[trip_i][3] == self.trip_list[duty[-1]][5]
                trip_depart_after = self.trip_list[trip_i][2] >= self.trip_list[duty[-1]][4]
                if trip_to_depot and trip_from_here and trip_depart_after:
                    trip_dur = self.minute_diff(self.trip_list[trip_i][2],self.trip_list[trip_i][4])
                    if trip_dur > trip_dur_last:
                        trip_dur_last = trip_dur
                        last_trip_i = trip_i
        diff_minutes = self.minute_diff(earliest_ot,latest_dt)

        return int(trip_dur_first + diff_minutes + trip_dur_last), first_trip_i, last_trip_i
    
    def get_break_dur_s(self, duty: list) -> int:
        total_break = 0
        for i in range(len(duty)-1):
            gap_dur = self.minute_diff(self.trip_list[duty[i]][4],self.trip_list[duty[i+1]][2])
            if gap_dur >= S_MIN_BREAK:
                total_break += gap_dur

        return int(total_break)

    def get_break_dur_l(self, duty: list) -> int:
        total_break = 0
        break_dur = 0
        for i in range(len(duty)-1):
            gap_dur = self.minute_diff(self.trip_list[duty[i]][4],self.trip_list[duty[i+1]][2])
            if gap_dur >= L_BREAK:
                total_break += gap_dur
                break_dur = 0
            elif break_dur>=L_BREAK1 and gap_dur>=L_BREAK2:
                total_break += (break_dur + gap_dur)
                break_dur = 0
            elif break_dur==0 and gap_dur>=L_BREAK1:
                break_dur += gap_dur

        return int(total_break)

    def get_break_dur(self, duty: list, is_short: bool, first_trip_i: int = None, last_trip_i: int = None) -> int:
        """
        is_short: duty type, True for short, False for long
        first_trip_i: connecting trip from depot; an index to a trip in self.trip_list
        last_trip_i: connecting trip to depot; an index to a trip in self.trip_list
        """
        total_break = 0

        first_trip_break_dur = 0
        last_trip_break_dur = 0
        if first_trip_i != None:
            if not (self.trip_list[first_trip_i][3]==self.trip_list[duty[0]][3] and
                    self.trip_list[first_trip_i][5]==self.trip_list[duty[0]][5]):
                first_trip_break_dur = self.minute_diff(self.trip_list[first_trip_i][2],
                                                        self.trip_list[duty[0]][2])
        if last_trip_i != None:
            if not (self.trip_list[last_trip_i][3]==self.trip_list[duty[-1]][3] and
                    self.trip_list[last_trip_i][5]==self.trip_list[duty[-1]][5]):
                last_trip_break_dur = self.minute_diff(self.trip_list[last_trip_i][4],
                                                       self.trip_list[duty[-1]][4])
        if is_short:
            total_break = self.get_break_dur_s(duty)
        else:
            total_break = self.get_break_dur_l(duty)
        total_break += first_trip_break_dur
        total_break += last_trip_break_dur

        return int(total_break)

    #########################################
    ### generate_schedule() sub-functions ###
    #########################################
    def build_break(self, trip_before: list, trip_after: list) -> list:
        # code trip, line, ot, op, dt, dp, distance, bus
        break_container = ['BREAK','-']
        break_container.append(trip_before[4])
        break_container.append(trip_before[5])
        break_container.append(trip_after[2])
        break_container.append(trip_after[3])
        break_container.extend([0.0,'-'])

        return break_container

    def triplist_to_duty(self, trip_list: list) -> list[int]:
        trip_codes = []
        for trip in trip_list:
            if not trip[0] in trip_codes:
                trip_codes.append(trip[0])
        trip_i_list = []
        for trip_i in range(len(self.trip_list)):
            if self.trip_list[trip_i][0] in trip_codes:
                trip_i_list.append(trip_i)
            
        return trip_i_list
    
    def get_gap_connections(self, trip_list: list) -> list:
        """Returns connections that are furthest to the left/right of the gap"""
        best_connections = []
        for trip_i in range(len(trip_list)-1):
            if trip_list[trip_i][5] != trip_list[trip_i+1][3]:
                if not TRAVEL_AS_PASSENGER:
                    print('Infeasible sol. Indirect trips found but TRAVEL_AS_PASSENGER is false!')
                assert TRAVEL_AS_PASSENGER==True
                all_connections: List[List[int,List]] = []
                for trip in self.trip_list:
                    is_connecting_trip = trip[3]==trip_list[trip_i][5] and trip[5]==trip_list[trip_i+1][3]
                    is_between = trip[2]>=trip_list[trip_i][4] and trip[4]<=trip_list[trip_i+1][2]
                    if is_connecting_trip and is_between:
                        left_idle_dur = self.minute_diff(trip_list[trip_i][4],trip[2])
                        right_idle_dur = self.minute_diff(trip[4],trip_list[trip_i+1][2])
                        left_right_diff = abs(left_idle_dur-right_idle_dur)
                        all_connections.append([left_right_diff,trip])
                best_connections.append(max(all_connections, key=self.take_lr_diff)[1])

        return best_connections
    
    #################
    ### UTILITIES ###
    #################
    def take_suffix(self, elem: str):
        splitted_str = elem.split('.')[-2].split('_')
        try:
            suffix = int(splitted_str[-1])
        except:
            suffix = 0 # i.e. non numerical suffix will be first
        return suffix
    
    def attach_date(self, trip_list: list) -> list:
        changed_trip_list = trip_list
        for i in range(len(changed_trip_list)):
            ot_datetime = datetime.datetime.combine(datetime.date.today(), changed_trip_list[i][2])
            dt_datetime = datetime.datetime.combine(datetime.date.today(), changed_trip_list[i][4])
            if ot_datetime >= dt_datetime:
                dt_datetime = dt_datetime + datetime.timedelta(days=1)
            changed_trip_list[i][2] = ot_datetime
            changed_trip_list[i][4] = dt_datetime
        return changed_trip_list
    
    def minute_diff(self, dtime_before: datetime.datetime, dtime_after: datetime.datetime) -> int:
        time_diff = dtime_after - dtime_before
        time_diff = round(time_diff.total_seconds()/60)

        return int(time_diff)
    
    def take_ot(self, elem):
        if isinstance(elem, int):
            return self.trip_list[elem][2]
        else:
            return elem[2]
    
    def take_dt(self, elem):
        if isinstance(elem, int):
            return self.trip_list[elem][4]
        else:
            return elem[4]
        
    def take_lr_diff(self, elem):
        return elem[0]
    
    def take_swapped_cost(self, elem):
        _,_,swapped_cost = elem
        return swapped_cost
    
    def trip_to_tripi(self, trip: list) -> int:
        matching_trip_i = None
        for trip_i in range(len(self.trip_list)):
            if self.trip_list[trip_i][0] == trip[0]:
                matching_trip_i = trip_i
                break
        assert matching_trip_i != None

        return matching_trip_i
    
    def duties_to_chromosome(self, duty_list: list[list[int]]) -> list[int]:
        chromosome = [None for _ in range(len(self.trip_list))]
        for duty_i in range(len(duty_list)):
            for trip_i in duty_list[duty_i]:
                chromosome[trip_i] = duty_i
        return chromosome
    
    def csv_to_duties(self):
        import csv
        file = open('all_duties.csv', 'r')
        self.all_duties = list(csv.reader(file, delimiter=','))
        file.close()
        for duty_i in range(len(self.all_duties)):
            converted_row = [int(item) for item in self.all_duties[duty_i]]
            self.all_duties[duty_i] = converted_row
        print('check last row', self.all_duties[-1])

    def duties_to_csv(self):
        import csv
        with open('all_duties.csv', 'w', newline='') as f:
            write = csv.writer(f)
            write.writerows(self.all_duties)



class MainWindow:
    def __init__(self) -> None:
        mock_thread = MockThread()
        mock_thread.input_dir_path = 'C:\\COMMON\\FIVERR\\javivi_sevilla\\CODE\\input'
        mock_thread.start_thread()
        print('END')

if __name__=="__main__":
    window = MainWindow()
