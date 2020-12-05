LOCAL_DIR='/Users/vtek/gnn_pathplanning/'
cd ../
echo "Map Noise Prob 0.2" >> results.txt
python main.py configs/dcp_onlineExpert.json --map_noise_prob 0.2 --mode test --log_anime --best_epoch --test_general --log_time_trained 1582034757  --nGraphFilterTaps 3   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20 --num_testset 1
cd utils
python visualize.py --name map_noise_prob_0.2 --failure_case True --nGraphFilterTaps 3 --type dcpOE --log_time_trained 1582034757 --caseId 00000 --local_dir $LOCAL_DIR
cd ../
echo "Map Shift Units 1" >> results.txt
python main.py configs/dcp_onlineExpert.json --map_shift_units 1 --mode test --log_anime --best_epoch --test_general --log_time_trained 1582034757  --nGraphFilterTaps 3   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20 --num_testset 1
cd utils
python visualize.py --name map_shift_units_1 --failure_case True --nGraphFilterTaps 3 --type dcpOE --log_time_trained 1582034757 --caseId 00000 --local_dir $LOCAL_DIR
cd ../
echo "Feature Noise Stdev 5.0" >> results.txt
python main.py configs/dcp_onlineExpert.json --feature_noise_std 5.0 --mode test --log_anime --best_epoch --test_general --log_time_trained 1582034757  --nGraphFilterTaps 3   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20 --num_testset 1
cd utils
python visualize.py --name feature_noise_stdev_5 --failure_case True --nGraphFilterTaps 3 --type dcpOE --log_time_trained 1582034757 --caseId 00000 --local_dir $LOCAL_DIR
cd ../
echo "Move Noise Stdev 10.0" >> results.txt
python main.py configs/dcp_onlineExpert.json --move_noise_std 10.0 --mode test --log_anime --best_epoch --test_general --log_time_trained 1582034757  --nGraphFilterTaps 3   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20 --num_testset 1
cd utils
python visualize.py --name move_noise_stdev_10 --failure_case True --nGraphFilterTaps 3 --type dcpOE --log_time_trained 1582034757 --caseId 00000 --local_dir $LOCAL_DIR
cd ../
echo "Comm Loss Param 20.0" >> results.txt
python main.py configs/dcp_onlineExpert.json --comm_dropout_param 20.0 --mode test --log_anime --best_epoch --test_general --log_time_trained 1582034757  --nGraphFilterTaps 3   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20 --num_testset 1
cd utils
python visualize.py --name comm_loss_param_20 --failure_case True --nGraphFilterTaps 3 --type dcpOE --log_time_trained 1582034757 --caseId 00000 --local_dir $LOCAL_DIR
cd ../
echo "Sybil Attack Count 8" >> results.txt
python main.py configs/dcp_onlineExpert.json --sybil_attack_count 8 --mode test --log_anime --best_epoch --test_general --log_time_trained 1582034757  --nGraphFilterTaps 3   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20 --num_testset 1
cd utils
python visualize.py --name sybil_attack_count_8 --failure_case True --nGraphFilterTaps 3 --type dcpOE --log_time_trained 1582034757 --caseId 00000 --local_dir $LOCAL_DIR
cd ../
echo "Rogue Agent Count 8" >> results.txt
python main.py configs/dcp_onlineExpert.json --rogue_agent_count 8 --mode test --log_anime --best_epoch --test_general --log_time_trained 1582034757  --nGraphFilterTaps 3   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20 --num_testset 1
cd utils
python visualize.py --name rogue_agent_count_8 --failure_case True --nGraphFilterTaps 3 --type dcpOE --log_time_trained 1582034757 --caseId 00000 --local_dir $LOCAL_DIR
cd ../