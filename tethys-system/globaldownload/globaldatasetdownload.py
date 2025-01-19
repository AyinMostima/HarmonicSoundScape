import pandas as pd
import os
import warnings
from maad import sound, util, rois
import joblib


#记得检查修改scikit-maad包源代码避免缺失值，修改如下方法如下：
#df_dataset['lat'] = pd.to_numeric(df_dataset['lat'], errors='coerce')
#df_dataset['lng'] = pd.to_numeric(df_dataset['lng'], errors='coerce')
#可选，替换 NaN 值
#df_dataset['lat'].fillna(0, inplace=True)
#df_dataset['lng'].fillna(0, inplace=True)

def grab_audio(path, audio_format='mp3'):
    filelist = []
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if name[-3:].casefold() == audio_format and name[:2] != '._':
                filelist.append(os.path.join(root, name))
    return filelist
#%%
XC_ROOTDIR = './data/'
XC_DIR = 'global_dataset'
data = [
    # Aquatic and Wetland Surface Birds
    ['Eurasian Wigeon', 'Mareca penelope'],
    ['Tundra Swan', 'Cygnus columbianus'],
    ['Mallard', 'Anas platyrhynchos'],
    ['Greater White-fronted Goose', 'Anser albifrons'],
    ['Common Goldeneye', 'Bucephala clangula'],
    ['Green-winged Teal', 'Anas crecca'],
    ['Green-winged Teal', 'Anas carolinensis'],
    ['Eurasian Coot', 'Fulica atra'],
    ['Eurasian Moorhen', 'Gallinula chloropus'],

    # Shoreline and Marsh Birds
    ['Eurasian Curlew', 'Numenius arquata'],
    ['Whimbrel', 'Numenius phaeopus'],

    # Grassland Ground Birds
    ['Olive-backed Pipit', 'Anthus hodgsoni'],
    ['Yellow-browed Bunting', 'Emberiza chrysophrys'],
    ['Yellow-billed Grosbeak', 'Eophona migratoria'],

    # Shrub Layer Birds
    ['Light-vented Bulbul', 'Pycnonotus sinensis'],
    ['Chinese Hwamei', 'Garrulax canorus'],
    ['Japanese Tit', 'Parus minor'],
    ['Silver-throated Tit', 'Aegithalos glaucogularis'],
    ['Chinese Blackbird', 'Turdus mandarinus'],
    ['Pale-legged Leaf Warbler', 'Phylloscopus tenellipes'],

    # Lower Canopy Birds
    ['Pale Thrush', 'Turdus pallidus'],
    ['Oriental Magpie', 'Pica serica'],

    # Upper Canopy and Aerial Birds
    ["Swinhoe's White-eye", 'Zosterops simplex']
]
#%%
df_species = pd.DataFrame(data,columns =['english name',
                                        'scientific name'])
gen = []
sp = []
for name in df_species['scientific name']:
    gen.append(name.rpartition(' ')[0])
    sp.append(name.rpartition(' ')[2])

df_query = pd.DataFrame()
df_query['param1'] = gen
df_query['param2'] = sp
# df_query['param3'] ='type:drumming'
# df_query['param4'] ='area:europe'
# df_query['param5 ='len:"5-120"'
# df_query['param6'] ='q:">C"'

# Get recordings metadata corresponding to the query
df_dataset= util.xc_multi_query(df_query,
                                format_time = False,
                                format_date = False,
                                verbose=True)

joblib.dump(df_dataset, 'df_dataset.joblib')
df_dataset.to_excel('df_dataset.xlsx')
df_dataset.to_pickle('df_dataset.pkl')

util.xc_download(df_dataset,
                rootdir = XC_ROOTDIR,
                dataset_name= XC_DIR,
                overwrite=True,
                save_csv= True,
                verbose = True)

import numba
# 加速的lambda函数
@numba.jit(nopython=True)
def extract_real_filename(filename):
    return filename.split('-')[0]
# 使用 numba 加速的 apply
df_dataset['realfilename'] = df_dataset['file-name'].apply(extract_real_filename)



joblib.dump(df_dataset, 'df_dataset.joblib')
df_dataset.to_excel('df_dataset.xlsx')
df_dataset.to_pickle('df_dataset.pkl')
# df_dataset = joblib.load('df_dataset.joblib')