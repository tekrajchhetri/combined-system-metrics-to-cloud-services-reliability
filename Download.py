# -*- coding: utf-8 -*-
# @Author  : Tek Raj Chhetri
# @Email   : tekraj.chhetri@sti2.at
# @Web     : http://tekrajchhetri.com/
# @File    : Download.py
# @Software: PyCharm

import urllib.parse
import requests
import pandas as pd
import json
import csv
from itertools import zip_longest
import urllib.parse
import requests
import re
import numpy as np
import copy
import shutil
import glob

def combine_small_file_to_big():
    """Merge multiple csv  into one based on header
        https://stackoverflow.com/questions/44791212/concatenating-multiple-csv-files-into-a-single-csv-with-the-same-header-python/44791368    
    """
    #import csv files from folder
    path = r'./'
    allFiles = glob.glob(path + "/*.csv")
    allFiles.sort()  # glob lacks reliable ordering, so impose your own if output order matters
    with open('combined_data.csv', 'wb') as outfile:
        for i, fname in enumerate(allFiles):
            with open(fname, 'rb') as infile:
                if i != 0:
                    infile.readline()  # Throw away header on all but first file
                # Block copy rest of file from input to output without parsing
                shutil.copyfileobj(infile, outfile)
                print(fname + " has been imported.")

def maketarget(equallendict):

    """Make Target Label
       1 - Fail
       0 - OK/ Good condition
    """
    dictwithTargetLabel = copy.deepcopy(equallendict)
    targetValue=[]
    failureThreshold=[
                    101, #cpu_utilization
                    1000000000, #memory_utilization
                    5000, #network_overhead
                    16089590032253278.0, #io utilization
                    38775016960.0, #bits_outputted
                    3189693620224.0, #bits_inputted
                    10, #smart_188
                    10, #smart_197
                    10, #smart_198
                    10, #smart_9
                    51, #smart1
                    10,#smart5 
                    10,#smart187 
                    10,#smart_7-
                    21,#smart_3
                    10,#smart_4
                    96,#smart_194
                    10#smart_199
                ]
    isFailure = [False]*len(failureThreshold)
    
    for i in np.arange(len(equallendict["time"])):
        currentValue = [float(equallendict["cpu_utilization"][i]),
                  float(equallendict["memory_utilization"][i]),
                  float(equallendict["network_overhead"][i]),
                  float(equallendict["io_utilization"][i]),
                  float(equallendict["bits_outputted"][i]),
                  float(equallendict["bits_inputted"][i]),
                  float(equallendict["smart_188"][i]),
                  float(equallendict["smart_197"][i]),
                  float(equallendict["smart_198"][i]),
                  float(equallendict["smart_9"][i]),
                  float(equallendict["smart_1"][i]),
                  float(equallendict["smart_5"][i]),
                  float(equallendict["smart_187"][i]),
                  float(equallendict["smart_7"][i]),
                  float(equallendict["smart_3"][i]),
                  float(equallendict["smart_4"][i]),
                  float(equallendict["smart_194"][i]),
                  float(equallendict["smart_199"][i])]
        
        whatisResult = [currentValue[0] > failureThreshold[0], 
                         currentValue[1] > failureThreshold[1],
                         currentValue[2] > failureThreshold[2],
                         currentValue[3] > failureThreshold[3],
                         currentValue[4] > failureThreshold[4], 
                         currentValue[5] > failureThreshold[5], 
                         currentValue[6] < failureThreshold[6],
                         currentValue[7] < failureThreshold[7],
                         currentValue[8] < failureThreshold[8],
                         currentValue[9] < failureThreshold[9],
                         currentValue[10] < failureThreshold[10],
                         currentValue[11] < failureThreshold[11],
                         currentValue[12] < failureThreshold[12],
                         currentValue[13] < failureThreshold[13],
                         currentValue[14] < failureThreshold[14],
                         currentValue[15] < failureThreshold[15],
                         currentValue[16] > failureThreshold[16],
                         currentValue[17] < failureThreshold[17] 
                        ]

        if True in whatisResult:
            targetValue.append(1)
        else:
            targetValue.append(0)
            
    dictwithTargetLabel["target"] = targetValue
    return dictwithTargetLabel

def make_equal_size(unequal_dict):
    equallendict={}
    for key in unequal_dict:
        emptyary=[np.nan]*len(unequal_dict["time"])
        lenvalue = len(unequal_dict[key])
        emptyary[:lenvalue]=unequal_dict[key]
        equallendict[key] = emptyary
        del emptyary
    return equallendict

def query_data(query_metric, end_time,start_time,resolution="2h",customtype=False):
    
    
    if customtype:
        query=query_metric
    else:
        query="{__name__=~'"+query_metric+"'}"
    url = 'http://127.0.0.1:9090/api/v1/query_range'
    response = requests.get(url,
                            params={'query': query,
                                    'start': start_time,
                                    'end': end_time,
                                    'step': resolution})
    print(urllib.parse.unquote(response.url))

    if response.status_code == 200:
        jsondata =  response.json()
        return jsondata["data"]["result"]
    else:
         return "Error occured while parsing data::Status: {0} Message: {1}".format(response.status_code, response.text)
    
def query(start_time, end_time):
#     data = query_data('node_memory_MemAvailable_bytes',start_time=start_time,
#                   end_time=end_time)
    dataCPU = query_data("{__name__='node_load5'}/{__name__='node_cpu_core_total'}*100",
                         start_time=start_time,
                         end_time=end_time,customtype=True)
    for cpuutilizaiton in dataCPU:
        cpuutilizaiton["metric"]["__name__"]="cpu_utilization"
    dataMemory =query_data("{__name__='node_memory_MemTotal_bytes'}-{__name__='node_memory_MemAvailable_bytes'}",
                           start_time=start_time,
                           end_time=end_time,customtype=True)
    for memoryutilizaiton in dataMemory:
        memoryutilizaiton["metric"]["__name__"]="memory_utilization"
        
    node_network_transmit_bytes_total =query_data("{__name__='node_network_transmit_bytes_total'}",
                           start_time=start_time,
                           end_time=end_time,customtype=True)
    
    node_network_receive_bytes_total =query_data("{__name__='node_network_receive_bytes_total'}",
                           start_time=start_time,
                           end_time=end_time,customtype=True)
    
    node_disk_io_time_seconds_total =query_data("{__name__='node_disk_io_time_seconds_total'}",
                           start_time=start_time,
                           end_time=end_time,customtype=True)
    
    node_disk_read_bytes_total =query_data("{__name__='node_disk_read_bytes_total'}",
                           start_time=start_time,
                           end_time=end_time,customtype=True)
    
    node_disk_written_bytes_total =query_data("{__name__='node_disk_written_bytes_total'}",
                           start_time=start_time,
                           end_time=end_time,customtype=True)
    
    smartmon_command_timeout_value =query_data("{__name__='smartmon_command_timeout_value'}",
                           start_time=start_time,
                           end_time=end_time,customtype=True)

    smartmon_current_pending_sector_value =query_data("{__name__='smartmon_current_pending_sector_value'}",
                           start_time=start_time,
                           end_time=end_time,customtype=True)

    smartmon_reallocated_sector_ct_value =query_data("{__name__='smartmon_reallocated_sector_ct_value'}",
                           start_time=start_time,
                           end_time=end_time,customtype=True)
    smartmon_offline_uncorrectable_value =query_data("{__name__='smartmon_offline_uncorrectable_value'}",
                           start_time=start_time,
                           end_time=end_time,customtype=True)
    smartmon_reported_uncorrect_value =query_data("{__name__='smartmon_reported_uncorrect_value'}",
                           start_time=start_time,
                           end_time=end_time,customtype=True)
    smartmon_raw_read_error_rate_value =query_data("{__name__='smartmon_raw_read_error_rate_value'}",
                           start_time=start_time,
                           end_time=end_time,customtype=True)
    smartmon_udma_crc_error_count_value =query_data("{__name__='smartmon_udma_crc_error_count_value'}",
                           start_time=start_time,
                           end_time=end_time,customtype=True)
    smartmon_spin_up_time_value =query_data("{__name__='smartmon_spin_up_time_value'}",
                           start_time=start_time,
                           end_time=end_time,customtype=True)
    smartmon_start_stop_count_value =query_data("{__name__='smartmon_start_stop_count_value'}",
                           start_time=start_time,
                           end_time=end_time,customtype=True)
    smartmon_seek_error_rate_value =query_data("{__name__='smartmon_seek_error_rate_value'}",
                           start_time=start_time,
                           end_time=end_time,customtype=True)
    smartmon_temperature_celsius_value =query_data("{__name__='smartmon_temperature_celsius_value'}",
                           start_time=start_time,
                           end_time=end_time,customtype=True)
    smartmon_power_cycle_count_valuesmartmon_spin_retry_count_value =query_data("{__name__='smartmon_power_cycle_count_valuesmartmon_spin_retry_count_value'}",
                           start_time=start_time,
                           end_time=end_time,customtype=True)
    smartmon_power_on_hours_value =query_data("{__name__='smartmon_power_on_hours_value'}",
                           start_time=start_time,
                           end_time=end_time,customtype=True)

        
        
 
    return [dataCPU,
            dataMemory,
            node_network_transmit_bytes_total,
            node_network_receive_bytes_total,
            node_disk_io_time_seconds_total,
            node_disk_read_bytes_total,    
            node_disk_written_bytes_total,    
            smartmon_command_timeout_value,
            smartmon_current_pending_sector_value,
            smartmon_reallocated_sector_ct_value,
            smartmon_offline_uncorrectable_value,
            smartmon_reported_uncorrect_value,
            smartmon_raw_read_error_rate_value,
            smartmon_udma_crc_error_count_value,
            smartmon_spin_up_time_value,
            smartmon_start_stop_count_value,
            smartmon_seek_error_rate_value,
            smartmon_temperature_celsius_value,
            smartmon_power_cycle_count_valuesmartmon_spin_retry_count_value,
            smartmon_power_on_hours_value
           ]

def prom_metric_to_dict(start_time, end_time):
    dataList = query(start_time, end_time)
    finaldata = {"time":[],"id":[]}
    for data in dataList:
        for prom_data in data:
            metric_name = prom_data['metric']["__name__"] 
            """ Reference metric scrape id based on parides package
                https://goettl79.github.io/parides/
            """
            id_keys = {"instance", "job", "name"}
            metric_meta = prom_data['metric']
            ids = {label: metric_meta[label] for label in id_keys if label in metric_meta}
            metric_scrape_id = ""
            for value in sorted(ids.values()):
                metric_scrape_id = metric_scrape_id + "{}_".format(value)
            metric_scrape_id = metric_scrape_id[:-1]
            for value in sorted(prom_data["values"]):
                finaldata["time"].append(value[0]) 
                finaldata["id"].append(metric_scrape_id)
                if metric_name not in finaldata.keys():
                    finaldata[metric_name] = [value[1]]
                else:
                    finaldata[metric_name].append(value[1])

    networkOverhead = list(np.array(finaldata['node_network_transmit_bytes_total'],dtype=float)+\
                           np.array(finaldata['node_network_receive_bytes_total'],dtype=float))
    finaldata["network_overhead"]=networkOverhead
    finaldata.pop("node_network_transmit_bytes_total")
    finaldata.pop("node_network_receive_bytes_total")
    oldkeys = list(sorted(finaldata.keys()))
    newkey = ['cpu_utilization',
             'id', 
             'memory_utilization', 
             'network_overhead', 
             'io_utilization',
             'bits_outputted', 
             'bits_inputted',
             'smart_188', 
             'smart_197', 
             'smart_198', 
             'smart_9', 
             'smart_1',
             'smart_5',
             'smart_187', 
             'smart_7', 
             'smart_3', 
             'smart_4',
             'smart_194',
             'smart_199',
             'time']

    for i in range(len(oldkeys)):
        finaldata[newkey[i]] = finaldata.pop(oldkeys[i])
    return finaldata

def write_csv( start_time,end_time):
    inputdictionary = prom_metric_to_dict(start_time,end_time)
    filtered_dict = {key:value for key,value in inputdictionary.items() if "time" not in key and "id" not in key}
    lengths = max([len(value) for key,value in filtered_dict.items()])
    filtered_dict["time"]=inputdictionary["time"][:lengths]
    filtered_dict["id"]=inputdictionary["id"][:lengths]
    equallendict = make_equal_size(filtered_dict)
    dict_with_target_label = maketarget(equallendict)
    filename="{}_{}.csv".format(''.join(start_time.split(":")[:2]), ''.join(end_time.split(":")[:2]))
    dictionary_values = list(dict_with_target_label.values())
    column_header = list(dict_with_target_label.keys())
    coumn_data = zip_longest(*dictionary_values, fillvalue = '')
    with open(filename, 'w', newline='') as csvfile:
        wr = csv.writer(csvfile)
        wr.writerow(column_header)
        wr.writerows(coumn_data)
    print("Done writing {} file".format(filename))
    return dict_with_target_label
start_time=[
 #specify start time in ISO format YYYY-MM-DDT..
]
end_time= [
     #specify end time in ISO format YYYY-MM-DDT..
]
for i in range(len(start_time)):
    write_csv(start_time[i],end_time[i])
# print("All Data Download Completed. Now merging.....")
# combine_small_file_to_big()
# print("Merge Completed.....")
