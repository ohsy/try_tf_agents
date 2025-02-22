
import os
import argparse
import pathlib


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="argpars parser used")
    parser.add_argument('-y', '--playground_index', type=int, help="playground index")
    parser.add_argument('-s', '--num_shellscripts', type=int, help="number of shell scripts")

    #--------------- from play.py
    parser.add_argument('-e', '--environment', type=str,
            choices=['CartPole-v0','Pendulum-v1','Pendulum-v1_discrete','Reacher-v2','Reacher-v2_discrete',\
                    'DaisoSokcho','DaisoSokcho_discrete','DaisoSokcho_discrete_unit1'])
    parser.add_argument('-w', '--environment_wrapper', type=str, choices=['history'],
            help="environment wrapper: 'history' adds observation and action history to the environment's observations.")
    parser.add_argument('-a', '--agent', type=str, choices=['DQN','DQN_multiagent','CDQN','CDQN_multiagent','DDPG','TD3','SAC','SAC_wo_normalize','BC','CQL_SAC','CQL_SAC_wo_normalize'])
    parser.add_argument('-r', '--replaybuffer', type=str, choices=['reverb','tf_uniform'], help="'reverb' must be used with driver 'py'")
    parser.add_argument('-d', '--driver', type=str, choices=['py','dynamic_step','dynamic_episode','none'])
    parser.add_argument('-c', '--checkpoint_path', type=str, help="to restore")
    parser.add_argument('-f', '--fill_after_restore', type=str, help="fill replaybuffer with agent.policy after restoring agent",
            choices=['true','false'])
    parser.add_argument('-p', '--reverb_checkpoint_path', type=str, help="to restore: parent directory of saved path," +
            " which is output when saved, like '/tmp/tmp6j63a_f_' of '/tmp/tmp6j63a_f_/2024-10-27T05:22:20.16401174+00:00'")
    parser.add_argument('-n', '--num_actions', type=int, help="number of actions for ActionDiscretizeWrapper")
    parser.add_argument('-t', '--num_time_steps', type=int, help="number of time-steps")
    parser.add_argument('-l', '--replaybuffer_max_length', type=int, help="replaybuffer max length")
    parser.add_argument('-i', '--num_env_steps_to_collect_init', type=int, help="number of initial collect steps")
    parser.add_argument('-g', '--epsilon_greedy', type=float, help="epsilon for epsilon_greedy")
    parser.add_argument('-o', '--reverb_port', type=int, help="reverb port for reverb.Client and Server")
    args = parser.parse_args()

    playground_index = args.playground_index
    num_shellscripts = args.num_shellscripts  # to get statistical behavior by averaging

    project_path = "/home/soh/work/try_tf_agents"
    src_path = "/home/soh/work/try_tf_agents/src"
    playground_path = os.path.join(project_path, f"playground/{args.environment}_{args.agent}_{playground_index}")
    checkpoint_path = os.path.join(project_path, args.checkpoint_path) if args.checkpoint_path is not None else None

    output_prefix = f"o" \
            + (f"_num_time_steps_{args.num_time_steps}" if args.num_time_steps is not None else "") \
            + (f"_replaybuffer_max_length_{args.replaybuffer_max_length}" if args.replaybuffer_max_length is not None else "") \
            + (f"_num_env_steps_to_collect_init_{args.num_env_steps_to_collect_init}" if args.num_env_steps_to_collect_init is not None else "")

    master_sh_path = os.path.join(playground_path, f"playbatch.sh")
    pathlib.Path(master_sh_path).parent.mkdir(exist_ok=True, parents=True)  # make parents unless they exist
    with open(master_sh_path,'w') as f:
        f.write(f"#!/bin/bash\n\n")
        for idx in range(num_shellscripts):
            f.write(f"./playbatch{idx}.sh\n")
    os.chmod(master_sh_path, 0o775)

    for idx in range(num_shellscripts):
        sub_sh_path = os.path.join(playground_path, f"playbatch{idx}.sh")
        with open(sub_sh_path,'w') as f:
            f.write(f"#!/bin/bash\n\n")
            f.write(f"cd {src_path}\n")
            instruction = f"python3 play.py" \
                    + f" -e {args.environment}" \
                    + f" -a {args.agent}" \
                    + (f" -t {args.num_time_steps}" if args.num_time_steps is not None else "") \
                    + (f" -l {args.replaybuffer_max_length}" if args.replaybuffer_max_length is not None else "") \
                    + (f" -i {args.num_env_steps_to_collect_init}" if args.num_env_steps_to_collect_init is not None else "") \
                    + (f" -c {checkpoint_path}" if args.checkpoint_path is not None else "")
            instruction += f" &> {playground_path}/{output_prefix}_{idx} &"
            f.write(f"{instruction}\n")
        os.chmod(sub_sh_path, 0o775)

    postprocess_sh_path = os.path.join(playground_path, f"postprocess.sh")
    with open(postprocess_sh_path,'w') as f:
        f.write(f"#!/bin/bash\n\n")
        f.write(f"cd {src_path}\n")
        f.write(f"python3 get_avg_returns.py {num_shellscripts} {playground_path}/{output_prefix}\n")
        f.write(f"python3 plot_csv.py {playground_path}/{output_prefix}\n")
        f.write(f"cd {playground_path}\n")
        f.write(f"mkdir {output_prefix}\n")
        f.write(f"mv {output_prefix}.* {output_prefix}\n")
        f.write(f"mv {output_prefix}_* {output_prefix}\n")
        f.write(f"cp {src_path}/play.py {src_path}/game.py {src_path}/config.json playbatch*.sh postprocess.sh {output_prefix}\n")
    os.chmod(postprocess_sh_path, 0o775)
