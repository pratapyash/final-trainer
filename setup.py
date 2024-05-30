import os
os.system('sudo apt-get update')
os.system('sudo apt-get install -y libsndfile1 ffmpeg')
# os.system('sudo apt-get install -y build-essential')
# os.system('sudo apt-get install -y gcc g++')
# os.system('sudo apt-get install -y libstdc++6')
os.system('pip install -r requirements.txt')
# os.system('pip install nemo_toolkit[all]')
# os.system('pip install nemo')