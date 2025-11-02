
import threading
import os
import subprocess
from pathlib import Path

"""
The following is parameters needed for training DNN
Model_name have values: SPF, NH, N04F, 
                        CN05F-10and25, CN06F-10and25,
                        CN10F-10and25, CN12F-10and25,
                        CN10F-50, CN12F-50, (These two is a little slow)
"""
Model_name='NH'
Regenerate_train_txt=False



"""
The following is subfunctions needed for training DNN
"""
def Prepare_SPF_training_data():
    current_path = os.getcwd()
    #matlab version here is R2022b, if your version is different, maybe above this version is better
    if Regenerate_train_txt==True or not os.path.exists('.\\SPF\\SPF_Land_Water_Training.txt'):
        os.system(f'matlab -batch \"cd(\'{current_path}\')\"')
        os.system('matlab -batch \"run(\'.\\SPF\\Prepare_training_data_SPF.m\')\"')

def Prepare_NH_training_data():
    current_path = os.getcwd()
    if Regenerate_train_txt==True or not os.path.exists('.\\NF\\NH_Land_Water_Training.txt'):
        os.system(f'matlab -batch \"cd(\'{current_path}\')\"')
        os.system('matlab -batch \"run(\'.\\NF\\Prepare_training_data_NH.m\')\"')

def Prepare_N04F_training_data():
    current_path = os.getcwd()
    if Regenerate_train_txt==True or not os.path.exists('.\\NF\\N04F_Land_Water_Training.txt'):
        os.system(f'matlab -batch \"cd(\'{current_path}\')\"')
        os.system('matlab -batch \"run(\'.\\NF\\Prepare_training_data_N04F.m\')\"')

def Prepare_CNF_training_data():
    current_path = os.getcwd()
    if Regenerate_train_txt==True or not os.path.exists('.\\CNF\\CNF_Land_Water_Training.txt'):
        os.system(f'matlab -batch \"cd(\'{current_path}\')\"')
        os.system('matlab -batch \"run(\'.\\CNF\\Prepare_training_data_CNF.m\')\"')

# model tasks
def train_SP05F():
    print("Start to train SP05F!")
    with subprocess.Popen(['python', '-u', '.\\SPF\\MLP_train_SPF.py',
                    '--featureNumbers', '3'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    bufsize=1
                    ) as process:
                        for line in process.stdout:
                            print(line, end='', flush=True)

def train_SP06F():
    print("Start to train SP06F!")
    with subprocess.Popen(['python', '-u', '.\\SPF\\MLP_train_SPF.py',
                    '--featureNumbers', '4'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    bufsize=1
                    ) as process:
                        for line in process.stdout:
                            print(line, end='', flush=True)

def train_SP10F():
    print("Start to train SP10F!")
    with subprocess.Popen(['python', '-u', '.\\SPF\\MLP_train_SPF.py',
                    '--featureNumbers', '8'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    bufsize=1
                    ) as process:
                        for line in process.stdout:
                            print(line, end='', flush=True)

def train_SP12F():
    print("Start to train SP12F!")
    with subprocess.Popen(['python', '-u', '.\\SPF\\MLP_train_SPF.py',
                    '--featureNumbers', '10'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    bufsize=1
                    ) as process:
                        for line in process.stdout:
                            print(line, end='', flush=True)

def train_NH_10():
    print("Start to train NH-10!")
    with subprocess.Popen(['python', '-u', '.\\NF\\MLP_NH_train.py',
                    '--KNN', '10'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    bufsize=1
                    ) as process:
                        for line in process.stdout:
                            print(line, end='', flush=True)

def train_NH_25():
    print("Start to train NH-25!")
    with subprocess.Popen(['python', '-u', '.\\NF\\MLP_NH_train.py',
                    '--KNN', '25'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    bufsize=1
                    ) as process:
                        for line in process.stdout:
                            print(line, end='', flush=True)

def train_NH_50():
    print("Start to train NH-50!")
    with subprocess.Popen(['python', '-u', '.\\NF\\MLP_NH_train.py',
                    '--KNN', '50'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    bufsize=1
                    ) as process:
                        for line in process.stdout:
                            print(line, end='', flush=True)

def train_N04F_10():
    print("Start to train N04F-10!")
    with subprocess.Popen(['python', '-u', '.\\NF\\MLP_N04F_train.py',
                    '--KNN', '10'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    bufsize=1
                    ) as process:
                        for line in process.stdout:
                            print(line, end='', flush=True)

def train_N04F_25():
    print("Start to train N04F-25!")
    with subprocess.Popen(['python', '-u', '.\\NF\\MLP_N04F_train.py',
                    '--KNN', '25'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    bufsize=1
                    ) as process:
                        for line in process.stdout:
                            print(line, end='', flush=True)

def train_N04F_50():
    print("Start to train N04F-50!")
    with subprocess.Popen(['python', '-u', '.\\NF\\MLP_N04F_train.py',
                    '--KNN', '50'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    bufsize=1
                    ) as process:
                        for line in process.stdout:
                            print(line, end='', flush=True)

def train_CNF(modelName):
    print(f"Start to train {modelName}!")
    if modelName=="CN05F-10":
        functionFile="DNN_CN05F_train.py"
        knnStr="10"
    elif modelName=="CN05F-25":
        functionFile="DNN_CN05F_train.py"
        knnStr="25"
    elif modelName=="CN06F-10":
        functionFile="DNN_CN06F_train.py"
        knnStr="10"
    elif modelName=="CN06F-25":
        functionFile="DNN_CN06F_train.py"
        knnStr="25"
    elif modelName=="CN10F-10":
        functionFile="DNN_CN10F_train.py"
        knnStr="10"
    elif modelName=="CN10F-25":
        functionFile="DNN_CN10F_train.py"
        knnStr="25"
    elif modelName=="CN10F-50":
        functionFile="DNN_CN10F_train.py"
        knnStr="50"
    elif modelName=="CN12F-10":
        functionFile="DNN_CN12F_train.py"
        knnStr="10"
    elif modelName=="CN12F-25":
        functionFile="DNN_CN12F_train.py"
        knnStr="25"
    elif modelName=="CN12F-50":
        functionFile="DNN_CN12F_train.py"
        knnStr="50"
    else:
        print("modelName Error in train_CNF()!")

    functionFilePath=".\\CNF\\"+functionFile
    with subprocess.Popen(['python', '-u', functionFilePath,
                    '--KNN', knnStr],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    bufsize=1
                    ) as process:
                        for line in process.stdout:
                            print(line, end='', flush=True)




"""
Main function of this python file
"""
if __name__ == "__main__":
    
    current_dir = Path(__file__).parent
    print(f'Current working path is {current_dir}')

    if(Model_name=='SPF'):
        Prepare_SPF_training_data()
        # Create thread
        t1 = threading.Thread(target=train_SP05F)
        t2 = threading.Thread(target=train_SP06F)
        t3 = threading.Thread(target=train_SP10F)
        t4 = threading.Thread(target=train_SP12F)

        # Start thread
        t1.start()
        t2.start()
        t3.start()
        t4.start()

        # waiting thread to end
        t1.join()
        t2.join()
        t3.join()
        t4.join()

    if(Model_name=='NH' or Model_name=='N04F'):        
        # Create thread
        if(Model_name=='NH'):
            Prepare_NH_training_data()
            t1 = threading.Thread(target=train_NH_10)
            t2 = threading.Thread(target=train_NH_25)
            t3 = threading.Thread(target=train_NH_50)
        if(Model_name=='N04F'):
            Prepare_N04F_training_data()
            t1 = threading.Thread(target=train_N04F_10)
            t2 = threading.Thread(target=train_N04F_25)
            t3 = threading.Thread(target=train_N04F_50)

        # Start thread
        t1.start()
        t2.start()
        t3.start()

        # waiting thread to end
        t1.join()
        t2.join()
        t3.join()

    if(Model_name=='CN05F-10and25' or Model_name=='CN06F-10and25'  or 
       Model_name=='CN10F-10and25' or Model_name=='CN12F-10and25'  or
       Model_name=='CN10F-50' or Model_name=='CN12F-50'):       
       Prepare_CNF_training_data()

    if(Model_name=='CN05F-10and25' or Model_name=='CN06F-10and25'  or 
       Model_name=='CN10F-10and25' or Model_name=='CN12F-10and25'):
        if(Model_name=='CN05F-10and25'):
            t1 = threading.Thread(target=train_CNF,args=("CN05F-10",))
            t2 = threading.Thread(target=train_CNF,args=("CN05F-25",))
        if(Model_name=='CN06F-10and25'):
            t1 = threading.Thread(target=train_CNF,args=("CN06F-10",))
            t2 = threading.Thread(target=train_CNF,args=("CN06F-25",))
        if(Model_name=='CN10F-10and25'):
            t1 = threading.Thread(target=train_CNF,args=("CN10F-10",))
            t2 = threading.Thread(target=train_CNF,args=("CN10F-25",))
        if(Model_name=='CN12F-10and25'):
            t1 = threading.Thread(target=train_CNF,args=("CN12F-10",))
            t2 = threading.Thread(target=train_CNF,args=("CN12F-25",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    if(Model_name=='CN10F-50' or Model_name=='CN12F-50'):
        if(Model_name=='CN10F-50'):
            t1 = threading.Thread(target=train_CNF,args=("CN10F-50",))
        if(Model_name=='CN12F-50'):
            t1 = threading.Thread(target=train_CNF,args=("CN12F-50",))
        t1.start()
        t1.join()






