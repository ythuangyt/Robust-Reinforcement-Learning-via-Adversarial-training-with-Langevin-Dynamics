import argparse
import os
import numpy as np
import ast

parser = argparse.ArgumentParser()
parser.add_argument("--script_name", default='submit.sh')
parser.add_argument("--logs_folder", default='./logs')
parser.add_argument("--job_name", default='')
parser.add_argument("--env", default="simple-v0")
parser.add_argument("--optimizer", default="SGLD")
parser.add_argument("--thermal_noise", nargs='+', default=["0"])
parser.add_argument("--lr", nargs='+', default=["0"])
parser.add_argument("--num_episodes", nargs='+', default=["0"])
parser.add_argument("--delta", nargs='+', default=['0'])
parser.add_argument("--alpha", type=float)
parser.add_argument('--two_player', type=ast.literal_eval)
parser.add_argument("--Kt", nargs='+', default=["10"])
parser.add_argument("--beta", nargs='+', default=["0.9"])
args = parser.parse_args()

# If submit script does not exist, create it
if not os.path.isfile(args.script_name):
    with open(args.script_name, 'w') as file:
        file.write(f'''#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=10000

./staskfarm ${{1}}\n''')

for s in range(5):
    for delta in args.delta:
        for lr in args.lr:  
            if(args.two_player):		
                folder_name = f'{args.env}/Two_Player'
            else:
                folder_name = f'{args.env}/One_player'

            if(args.optimizer == "RMSprop"):
                experiment = f'delta_{delta}/RMSprop/lr_{lr}/{s}'
    
                path = f'{folder_name}/{experiment}'
    
                if not os.path.isdir(f'{args.logs_folder}/{path}'):
                    os.makedirs(f'{args.logs_folder}/{path}')
                                   
                print(path)
               
                command = f'python3.6 main.py --env_name {args.env} --lr {lr} --delta {delta} --thermal_noise 0 --optimizer RMSprop --alpha {args.alpha} --seed {s+123} --two_player {args.two_player}'
    
                experiment_path = f'{args.logs_folder}/{path}/command.txt'
                                   
                with open(experiment_path, 'w') as file:
                    file.write(f'{command}\n')
                                    
                print(command)
    
                if not args.job_name:
                    job_name = path
                else:
                    job_name = args.job_name
    
                os.system(f'sbatch --job-name={job_name} {args.script_name} {experiment_path}')
            else:
                for thermal_noise in (args.thermal_noise):             
                    for beta in (args.beta):
                        for Kt in (args.Kt):					
                            experiment = f'delta_{delta}/{args.optimizer}_thermal_{thermal_noise}/lr_{lr}/beta_{beta}/Kt_{Kt}/{s}'
            
                            path = f'{folder_name}/{experiment}'
            
                            if not os.path.isdir(f'{args.logs_folder}/{path}'):
                                os.makedirs(f'{args.logs_folder}/{path}')
                                           
                            print(path)
                       
                            command = f'python3.6 main.py --env_name {args.env} --lr {lr} --delta {delta} --thermal_noise {thermal_noise} --optimizer {args.optimizer} --Kt {Kt} --beta {beta} --alpha {args.alpha} --seed {s+123} --two_player {args.two_player}'
            
                            experiment_path = f'{args.logs_folder}/{path}/command.txt'
                                           
                            with open(experiment_path, 'w') as file:
                                file.write(f'{command}\n')
                                            
                            print(command)
            
                            if not args.job_name:
                                job_name = path
                            else:
                                job_name = args.job_name
            
                            os.system(f'sbatch --job-name={job_name} {args.script_name} {experiment_path}')

