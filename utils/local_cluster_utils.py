import os
import sys
from threading import Thread
import time
import subprocess
import fileinput


RAY_INSTALLED = False
try:
    import ray
    RAY_INSTALLED = True
except Exception as e:
    print('Ray not installed')


if RAY_INSTALLED:
    try:
        ray.init(address="auto")
    except Exception as e:
        print("No existing ray cluster found. run `launch`")




user = os.path.expanduser('~')
user_name = user.split('/')[-1]



head_node = "127.0.0.1"
head_port = 6379



def update_bashrc():
    os.system(f"cp custom_bashrc.sh ~/.bashrc")


def cluster_reboot():
    print(f"Rebooting..")
    os.system(f"sudo reboot")


def increase_file_limit():
    def check_ulimit(ip, limit):

        command = f"ulimit -n"

        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        stdout, stderr = process.communicate()

        return process.returncode == 0 and int(stdout.strip()) == limit

    limit = 100000

    with open("/etc/security/limits.conf", "r") as f:
        contents = f.readlines()

    values = [f"{user_name}            hard    nofile            {limit}\n",
            f"{user_name}            soft    nofile            {limit}\n",
            f"root            hard    nofile            {limit}\n",
            f"root            soft    nofile            {limit}\n"]

    for value in values:
        contents.insert(-4, value)


    with open("/tmp/limits.conf", 'w') as f:
        contents = "".join(contents)
        f.write(contents)

    os.system(f"sudo mv /tmp/limits.conf /etc/security/limits.conf")

    return True



def install():
    

    print("Downloading Ray")
    os.system("bash download_ray.sh")    

    print("Installing dependencies")

    os.system("sudo bash installer.sh")

    os.system("source ~/.bashrc; ~/miniconda/bin/conda init")
    os.system("source ~/.bashrc; conda create -y -n osdi24 python=3.6.10")
    os.system("source ~/.bashrc; conda activate osdi24; python3 -m pip install -r requirements.txt")
    print("Configuring ray")
    os.system("source ~/.bashrc; conda activate osdi24; python3 ray_patch/python/ray/setup-dev.py -y")

    print("Increasing ulimit")
    success = increase_file_limit()

    print("Rebooting!")
    cluster_reboot()



def get_ray_path():
    # Run the command and capture its output

    command = f"which ray"

    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    stdout, stderr = process.communicate()

    if process.returncode == 0:
        return stdout.strip()
    return None


def tear_down():


        
    ray_dir = get_ray_path()


    if not ray_dir:
        print("Ray not found")
        return

    os.system(f"{ray_dir} stop")



def launch():

        

    ray_dir = get_ray_path()


    if not ray_dir:
        print("Ray not found")
        return


    os.system(f"{ray_dir} stop")
    os.system(f"{ray_dir} start --head --node-ip-address={head_node} --port={head_port}")
    return

def ray_smoke_test():
    @ray.remote
    def sleep_on_each_core():
        import tensorflow as tf
        import socket

        time.sleep(5)

        conda_env_name = os.environ.get('CONDA_DEFAULT_ENV')

        return {"conda": conda_env_name,
                "tf_version": tf.__version__,
                "ray_version": ray.__version__,
                "socket": socket.gethostname()}

    cores = int(ray.cluster_resources().get('CPU'))

    futures = [sleep_on_each_core.remote() for _ in range(cores)]

    print(len(futures))
    result = ray.get(futures)
    print(len(result) == cores)
    print(result)

    conda_env_names = [res.get("conda") for res in result]
    tf_versions = [res.get("tf_version") for res in result]

    non_none = len(list(filter(lambda c: c != None, conda_env_names)))

    print(f"{non_none}/{cores} non none condas")




def get_status():


    ray_nodes = list(filter(lambda n: n["alive"], ray.nodes()))
    print("Num of nodes: %d" % len(ray_nodes))
    print(ray.cluster_resources())
    print(ray.available_resources())




if __name__ == '__main__':

    if len(sys.argv) != 2:
        print(f"usage: python3 cluster_utils.py install/launch")
    else:

        try:
            val = globals()[sys.argv[1]]()
            if val:
                print(val)
        except Exception as e:
            print("Error running command")
            raise e
