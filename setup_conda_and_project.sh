mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh

wget https://cs6035.s3.amazonaws.com/ML/env.yml

bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/machine/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/machine/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/machine/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/machine/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

/home/machine/miniconda3/bin/conda init

echo 'Set the remote read timeout to three minutes, choose longer if it still times out'
/home/machine/miniconda3/bin/conda config --set remote_read_timeout_secs 180.0

echo 'Now installing the cs6035_ML environment from the env.yml file'
/home/machine/miniconda3/bin/conda env create -f env.yml

# now we'll get the project files
wget https://cs6035.s3.amazonaws.com/ML/Student_Local_Testing.zip

mkdir Student_Local_Testing
mv Student_Local_Testing.zip Student_Local_Testing
cd Student_Local_Testing

unzip Student_Local_Testing.zip

echo ''
echo ''
echo '**********************************************************************************'
echo 'Now that your environment is installed ****YOU NEED TO OPEN A NEW WINDOW/SHELL****'
echo '                                         *** This reads the changed .bashrc file ***'
echo ''
echo 'Once you open the new window the prompt should start with (base)'
echo ''
echo 'You can activate the cs6035_ML environment with: conda activate cs6035_ML'
echo 'Next you need to activate the new conda env in your favorite IDE'
echo ''
