#!/bin/bash


if [ $# -lt 2 ]; then
    echo "Usage: $0 num_nodes user_name"
    exit 1
fi

# num_nodes
N=$1
USER=$2


echo $N
echo $USER


cp ~/.ssh/authorized_keys copy_authorized_keys
cat ~/.ssh/authorized_keys > authorized_keys


# create N keys
for i in $(seq 0 $((N-1)))
do
    ssh-keygen -f k$i -N ""
    cat k$i.pub >> authorized_keys
done

#create config
rm config

cat /users/$USER/.ssh/config >> config
echo "" >> config

for i in $(seq 0 $((N-1)))
do
    echo "" >> config
    echo $"Host 10.1.1.$(( 2+i ))" >> config
    echo $" HostName 10.1.1.$(( 2+i ))" >> config
    echo $" IdentityFile ~/.ssh/key" >> config
    echo $" StrictHostKeyChecking no" >> config
    
done

#disseminate to all nodes
for i in $(seq 0 $((N-1)))
do
    cat k$i | sudo ssh n$i "cat > /users/${USER}/.ssh/key"
    cat k$i.pub | sudo ssh n$i "cat > /users/${USER}/.ssh/key.pub"
    cat authorized_keys | sudo ssh n$i "cat > /users/${USER}/.ssh/authorized_keys"
    cat config | sudo ssh n$i "cat > /users/${USER}/.ssh/config"
done


rm k*
rm authorized_keys
rm config
