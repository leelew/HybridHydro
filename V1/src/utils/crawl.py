# ==============================================================================
# Purpose:
#   Download Near-Real-Time (NRT) SMAP L4 soil moisture and CLDAS forcing data 
# 
# author: Lu Li, 8/9/2021
# ==============================================================================

import os

import numpy as np
import requests
import datetime
import time


# ------------------------------------------------------------------------------
# Generate online path of SMAP L4 data.
# ------------------------------------------------------------------------------
def get_SMAP_L4_path(year, month, day, hour):

    # host url of SMAP
    HOST = 'https://n5eil01u.ecs.nsidc.org'

    # 005 for L4, 004 for L3
    VERSION = '.005'
    EDITION = '3000_Vv5030_001'

    # get foldername
    url_path = '{host}/SMAP/SPL4SMGP{version}/{year}.{month:02}.{day:02}/'.\
        format(host=HOST,
               version=VERSION,
               year=year,
               month=month,
               day=day)

    # get filename
    filename = 'SMAP_L4_SM_gph_{year}{month:02}{day:02}T{hour:02}{edition}.h5'.\
        format(year=year,
               month=month,
               day=day,
               hour=hour,
               edition=EDITION)

    # get url for SMAP L4
    url_SMAP_L4 = url_path + filename

    return url_SMAP_L4, filename


# ------------------------------------------------------------------------------
# download NRT SMAP L4 data near-real-time (~4-day before to ~2-day before).
# ------------------------------------------------------------------------------
def download_SMAP_L4_NRT():

    # --------------------------------------------------------------------------
    # Set username, password, and save path TODO: using parser 
    # --------------------------------------------------------------------------
    DATA_DIR_L4 = "/hard/lilu/SMAP_L4/SMAP_L4/"

    USERNAME = 'sysulewlee1@gmail.com'
    PASSWORD = '941313Li'

    # get real local time
    # --------------------
    local_time = time.localtime()
    year = local_time.tm_year
    month = local_time.tm_mon
    day = local_time.tm_mday

    # get SMAP data 5-day before to nowadays (if have)
    # ------------------------------------------------

    # Use a requests session to keep track of authentication credentials
    with requests.Session() as session:

        # auth
        session.auth = (USERNAME, PASSWORD)

        # get data
        for i in range(66, 30, -1):
            print(i)
            
            # get datetime for ~5-1 day before
            dt = datetime.datetime.now() - datetime.timedelta(days = i)

            # get data for corresponding datetime
            for hour in np.arange(1, 24, 3):
                print('Downloading SMAP L4 data for: ' +
                        str(dt.year) + '-' +
                        str(dt.month).zfill(2)+'-' +
                        str(dt.day).zfill(2)+'-' +
                        str(hour).zfill(2))
                
                # get online path for SMAP L4
                url_SMAP_L4, file_name_L4 = get_SMAP_L4_path(
                        dt.year, dt.month, dt.day, hour)

                # mkdir if don't exist path for save
                # -----------------------------------
                if not os.path.exists(DATA_DIR_L4):
                    os.mkdir(DATA_DIR_L4)

                folder = DATA_DIR_L4 + '{year}.{month:02}.{day:02}/'.format(
                    year=dt.year,
                    month=dt.month,
                    day=dt.day)

                if not os.path.exists(folder):
                    os.mkdir(folder)

                print(folder)
                print(file_name_L4)
                print(os.path.join(folder, file_name_L4))
                print(url_SMAP_L4)
                # judge if already exist hour file
                if not os.path.exists(os.path.join(folder, file_name_L4)):
                    response = session.get(url_SMAP_L4)

                    # If 401 response, retry to get 
                    if response.status_code == 401:
                        response = session.get(response.url)

                    # If haven't down crawl, show reason
                    assert response.ok, \
                        'Problem downloading data! Reason: {}'.\
                            format(response.reason)

                    # save file
                    with open(os.path.join(folder, file_name_L4), 'wb') as f:
                        f.write(response.content)

                    print('*** SMAP L4 data saved *** ')
                else:
                    print('already exist file')
                    
             
def get_SMAP_L3_path(year, month, day):
    # host url of SMAP
    HOST = 'https://n5eil01u.ecs.nsidc.org'
    # 005 for L4, 004 for L3
    VERSION = '.007'
    # get foldername
    url_path = '{host}/SMAP/SPL3SMP{version}/{year}.{month:02}.{day:02}/'.\
        format(host=HOST,
               version=VERSION,
               year=year,
               month=month,
               day=day)
    # get filename
    filename = 'SMAP_L3_SM_P_{year}{month:02}{day:02}_R17030_001.h5'.\
        format(year=year,
               month=month,
               day=day)
    # get url for SMAP L3
    url_SMAP_L3 = url_path + filename

    return url_SMAP_L3, filename

"""
print('Downloading SMAP L3 data for: '+str(year) +
      '-'+str(month).zfill(2)+'-'+str(day).zfill(2))

url_SMAP_L3, file_name_L3 = get_SMAP_L3_path(
    year, month, day)
out_path_L3 = os.path.join(DATA_DIR_L3, file_name_L3)
print(url_SMAP_L3)
response = session.get(url_SMAP_L3, headers={
                    'Connection': 'close'})


# If the response code is 401, we still need to authorize with earthdata.
if response.status_code == 401:
    response = session.get(response.url)
assert response.ok, 'Problem downloading data! Reason: {}'.\
    format(response.reason)

with open(out_path_L3, 'wb') as f:
    f.write(response.content)
print('*** SMAP L3 data saved to: ' + out_path_L3 + ' *** ')
"""


if __name__ == '__main__':
    download_SMAP_L4_NRT()
