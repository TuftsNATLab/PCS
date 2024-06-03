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


with open("/var/emulab/boot/hostmap") as fp:
    N, *_ = fp.readlines()
N = int(N.rstrip())
list_of_nodes = ["10.1.1.%d" % (2+node_id) for node_id in list(range(N))]

head_node, *worker_nodes = list_of_nodes[:]

head_port = 6379

def setup_keys():
    def check_ssh(ip):


        command = f"ssh -o BatchMode=yes {ip} exit"

        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        stdout, stderr = process.communicate()

        return process.returncode == 0


    os.system(f'bash setup_keys.sh {N} {user_name}')
    time.sleep(1)
    

    successful = True

    for nodes in list_of_nodes:
        if check_ssh(nodes):
            print(f"SSH connection to {nodes} successful")
        else:
            print(f"SSH connection to {nodes} failed")
            successful = False

    return successful



def configure_ray(exclude_head=False):


    run_on_nodes("conda activate osdi24; cd ~/PCS/utils; python3 ray_patch/python/ray/setup-dev.py -y")
    return

    nodes_to_configure = list_of_nodes[1:] if exclude_head else list_of_nodes[:]

    threads = list()
    for node in nodes_to_configure:
        

        t = Thread(target=os.system, args=(f"ssh {node} <configure_ray.sh",))
        t.start()
        threads.append(t)


    for t in threads:
        t.join()



def rsync():
    for node in worker_nodes:
        os.system(f"rsync -av {user}/PCS {node}:{user}/")


def update_bashrc():
    
    for node in list_of_nodes:
        os.system(f"ssh {node} 'cp ~/PCS/utils/custom_bashrc.sh ~/.bashrc'")

def installer(exclude_head=False):


    nodes_to_install = list_of_nodes[1:] if exclude_head else list_of_nodes[:]

    threads = list()
    for node in list_of_nodes:
        

        t = Thread(target=os.system, args=(f"ssh {node} <installer.sh",))
        t.start()
        threads.append(t)


    for t in threads:
        t.join()


    with open("/tmp/state", "w") as f:
        f.write("dependencies_installed")

    print("installation done!")



def cluster_reboot():
    print(f"Rebooting worker nodes")
    for node in worker_nodes:
        os.system(f"ssh {node} sudo reboot")
    print(f"Rebooting head node")
    os.system(f"sudo reboot")


def increase_file_limit():
    def check_ulimit(ip, limit):


        command = f"ssh {ip} 'ulimit -n'"

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


    for node in worker_nodes:
        os.system(f"sudo rsync -e 'ssh -o StrictHostKeyChecking=no' /etc/security/limits.conf {node}:/etc/security/")


    success = True

    for node in list_of_nodes:
        if check_ulimit(node, limit):
            print(f"successfully set ulimit -n to {limit} on {node}")
        else:
            print(f"ulimit -n is NOT {limit} on {node}")
            success = False

    return success



def run_on_nodes(cmd):

    threads = list()
    for node in list_of_nodes:
        

        t = Thread(target=os.system, args=(f"ssh {node} 'source ~/.bashrc; {cmd}'",))
        t.start()
        threads.append(t)


    for t in threads:
        t.join()



def install():
    


    print("Setting up keys to ssh between cluster nodes")
    success = setup_keys()
    

    if not success:
        print("Failed to set up keys. Exiting!")
        sys.exit(1)

    print("Downloading Ray")
    os.system("bash download_ray.sh")    


    print("Syncing cluster files")
    rsync()
    print("Changing bashrc at every node")
    update_bashrc()


    print("Installing dependencies")

    installer()


    print("Activating conda")
    run_on_nodes("~/miniconda/bin/conda init")
    run_on_nodes("conda create -y -n osdi24 python=3.6.10")
    run_on_nodes("echo conda activate osdi24 >> ~/.bashrc")
    run_on_nodes("conda activate osdi24; cd ~/PCS/utils; python3 -m pip install -r requirements.txt")
    
    print("Configuring ray")
    configure_ray()

    print("Increasing ulimit")
    success = increase_file_limit()

    if not success:
        print("Failed to set up ulimit. This may result in erroneous behaviour when running ray")
        sys.exit(1)

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

    # stopping all previous instances
    for node in list_of_nodes:
        os.system(f"ssh {node} {ray_dir} stop")



def launch():

        

    ray_dir = get_ray_path()


    if not ray_dir:
        print("Ray not found")
        return


    # stopping all previous instances
    for node in list_of_nodes:
        os.system(f"ssh {node} {ray_dir} stop")

    # starting on head node
    os.system(f"{ray_dir} start --head --node-ip-address={head_node} --port={head_port}")


    for node in worker_nodes:
        time.sleep(1)
        os.system(f"ssh {node} {ray_dir} start --address={head_node}:{head_port}")

    time.sleep(5)

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