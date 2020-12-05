cd ../

# Baseline
echo "Baseline: \n" >> results.txt
python main.py configs/dcp_ECBS.json --mode test --log_anime --best_epoch --test_general --log_time_trained 1582029525  --nGraphFilterTaps 1   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
echo "\n" >> results.txt
python main.py configs/dcp_ECBS.json --mode test --log_anime --best_epoch --test_general --log_time_trained 1582028194  --nGraphFilterTaps 2   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
echo "\n" >> results.txt
python main.py configs/dcp_ECBS.json --mode test --log_anime --best_epoch --test_general --log_time_trained 1582028876  --nGraphFilterTaps 3   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
echo "\n" >> results.txt
python main.py configs/dcp_onlineExpert.json --mode test --log_anime --best_epoch --test_general --log_time_trained 1582314635 --nGraphFilterTaps 2   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
echo "\n" >> results.txt
python main.py configs/dcp_onlineExpert.json --mode test --log_anime --best_epoch --test_general --log_time_trained 1582034757  --nGraphFilterTaps 3   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
echo "\n" >> results.txt
echo "Map Noise Prob: \n" >> results.txt
for i in 0.001 0.01 0.05 0.1 0.2
do
    echo "Param = $i\n" >> results.txt
    python main.py configs/dcp_ECBS.json --map_noise_prob $i --mode test --log_anime --best_epoch --test_general --log_time_trained 1582029525  --nGraphFilterTaps 1   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
    echo "\n" >> results.txt
    python main.py configs/dcp_ECBS.json --map_noise_prob $i --mode test --log_anime --best_epoch --test_general --log_time_trained 1582028194  --nGraphFilterTaps 2   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
    echo "\n" >> results.txt
    python main.py configs/dcp_ECBS.json --map_noise_prob $i --mode test --log_anime --best_epoch --test_general --log_time_trained 1582028876  --nGraphFilterTaps 3   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
    echo "\n" >> results.txt
    python main.py configs/dcp_onlineExpert.json --map_noise_prob $i --mode test --log_anime --best_epoch --test_general --log_time_trained 1582314635 --nGraphFilterTaps 2   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
    echo "\n" >> results.txt
    python main.py configs/dcp_onlineExpert.json --map_noise_prob $i --mode test --log_anime --best_epoch --test_general --log_time_trained 1582034757  --nGraphFilterTaps 3   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
    echo "\n" >> results.txt
done
echo "Map Shift Units: \n" >> results.txt
for i in 1 2 3
do
    echo "Param = $i\n" >> results.txt
    python main.py configs/dcp_ECBS.json --map_shift_units $i --mode test --log_anime --best_epoch --test_general --log_time_trained 1582029525  --nGraphFilterTaps 1   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
    echo "\n" >> results.txt
    python main.py configs/dcp_ECBS.json --map_shift_units $i --mode test --log_anime --best_epoch --test_general --log_time_trained 1582028194  --nGraphFilterTaps 2   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
    echo "\n" >> results.txt
    python main.py configs/dcp_ECBS.json --map_shift_units $i --mode test --log_anime --best_epoch --test_general --log_time_trained 1582028876  --nGraphFilterTaps 3   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
    echo "\n" >> results.txt
    python main.py configs/dcp_onlineExpert.json --map_shift_units $i --mode test --log_anime --best_epoch --test_general --log_time_trained 1582314635 --nGraphFilterTaps 2   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
    echo "\n" >> results.txt
    python main.py configs/dcp_onlineExpert.json --map_shift_units $i --mode test --log_anime --best_epoch --test_general --log_time_trained 1582034757  --nGraphFilterTaps 3   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
    echo "\n" >> results.txt
done
echo "Feature Noise Std: \n" >> results.txt
for i in 0.1 0.2 0.5 0.75 1.0 5.0
do
    echo "Param = $i\n" >> results.txt
    python main.py configs/dcp_ECBS.json --feature_noise_std $i --mode test --log_anime --best_epoch --test_general --log_time_trained 1582029525  --nGraphFilterTaps 1   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
    echo "\n" >> results.txt
    python main.py configs/dcp_ECBS.json --feature_noise_std $i --mode test --log_anime --best_epoch --test_general --log_time_trained 1582028194  --nGraphFilterTaps 2   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
    echo "\n" >> results.txt
    python main.py configs/dcp_ECBS.json --feature_noise_std $i --mode test --log_anime --best_epoch --test_general --log_time_trained 1582028876  --nGraphFilterTaps 3   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
    echo "\n" >> results.txt
    python main.py configs/dcp_onlineExpert.json --feature_noise_std $i --mode test --log_anime --best_epoch --test_general --log_time_trained 1582314635 --nGraphFilterTaps 2   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
    echo "\n" >> results.txt
    python main.py configs/dcp_onlineExpert.json --feature_noise_std $i --mode test --log_anime --best_epoch --test_general --log_time_trained 1582034757  --nGraphFilterTaps 3   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
    echo "\n" >> results.txt
done
echo "Move Noise Std: \n" >> results.txt
for i in 0.5 1.0 2.0 5.0 10.0
do
    echo "Param = $i\n" >> results.txt
    python main.py configs/dcp_ECBS.json --move_noise_std $i --mode test --log_anime --best_epoch --test_general --log_time_trained 1582029525  --nGraphFilterTaps 1   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
    echo "\n" >> results.txt
    python main.py configs/dcp_ECBS.json --move_noise_std $i --mode test --log_anime --best_epoch --test_general --log_time_trained 1582028194  --nGraphFilterTaps 2   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
    echo "\n" >> results.txt
    python main.py configs/dcp_ECBS.json --move_noise_std $i --mode test --log_anime --best_epoch --test_general --log_time_trained 1582028876  --nGraphFilterTaps 3   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
    echo "\n" >> results.txt
    python main.py configs/dcp_onlineExpert.json --move_noise_std $i --mode test --log_anime --best_epoch --test_general --log_time_trained 1582314635 --nGraphFilterTaps 2   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
    echo "\n" >> results.txt
    python main.py configs/dcp_onlineExpert.json --move_noise_std $i --mode test --log_anime --best_epoch --test_general --log_time_trained 1582034757  --nGraphFilterTaps 3   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
    echo "\n" >> results.txt
done
echo "Comm Dropout Param: \n" >> results.txt
for i in 0.1 0.5 1.0 5.0 20.0
do
    echo "Param = $i\n" >> results.txt
    python main.py configs/dcp_ECBS.json --comm_dropout_param $i --mode test --log_anime --best_epoch --test_general --log_time_trained 1582029525  --nGraphFilterTaps 1   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
    echo "\n" >> results.txt
    python main.py configs/dcp_ECBS.json --comm_dropout_param $i --mode test --log_anime --best_epoch --test_general --log_time_trained 1582028194  --nGraphFilterTaps 2   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
    echo "\n" >> results.txt
    python main.py configs/dcp_ECBS.json --comm_dropout_param $i --mode test --log_anime --best_epoch --test_general --log_time_trained 1582028876  --nGraphFilterTaps 3   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
    echo "\n" >> results.txt
    python main.py configs/dcp_onlineExpert.json --comm_dropout_param $i --mode test --log_anime --best_epoch --test_general --log_time_trained 1582314635 --nGraphFilterTaps 2   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
    echo "\n" >> results.txt
    python main.py configs/dcp_onlineExpert.json --comm_dropout_param $i --mode test --log_anime --best_epoch --test_general --log_time_trained 1582034757  --nGraphFilterTaps 3   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
    echo "\n" >> results.txt
done
echo "Sybil Attack Count: \n" >> results.txt
for i in 1 2 3 5 8
do
    echo "Param = $i\n" >> results.txt
    python main.py configs/dcp_ECBS.json --sybil_attack_count $i --mode test --log_anime --best_epoch --test_general --log_time_trained 1582029525  --nGraphFilterTaps 1   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
    echo "\n" >> results.txt
    python main.py configs/dcp_ECBS.json --sybil_attack_count $i --mode test --log_anime --best_epoch --test_general --log_time_trained 1582028194  --nGraphFilterTaps 2   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
    echo "\n" >> results.txt
    python main.py configs/dcp_ECBS.json --sybil_attack_count $i --mode test --log_anime --best_epoch --test_general --log_time_trained 1582028876  --nGraphFilterTaps 3   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
    echo "\n" >> results.txt
    python main.py configs/dcp_onlineExpert.json --sybil_attack_count $i --mode test --log_anime --best_epoch --test_general --log_time_trained 1582314635 --nGraphFilterTaps 2   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
    echo "\n" >> results.txt
    python main.py configs/dcp_onlineExpert.json --sybil_attack_count $i --mode test --log_anime --best_epoch --test_general --log_time_trained 1582034757  --nGraphFilterTaps 3   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
    echo "\n" >> results.txt
done
echo "Rogue Agent Count: \n" >> results.txt
for i in 1 2 3 5 8
do
    echo "Param = $i\n" >> results.txt
    python main.py configs/dcp_ECBS.json --rogue_agent_count $i --mode test --log_anime --best_epoch --test_general --log_time_trained 1582029525  --nGraphFilterTaps 1   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
    echo "\n" >> results.txt
    python main.py configs/dcp_ECBS.json --rogue_agent_count $i --mode test --log_anime --best_epoch --test_general --log_time_trained 1582028194  --nGraphFilterTaps 2   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
    echo "\n" >> results.txt
    python main.py configs/dcp_ECBS.json --rogue_agent_count $i --mode test --log_anime --best_epoch --test_general --log_time_trained 1582028876  --nGraphFilterTaps 3   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
    echo "\n" >> results.txt
    python main.py configs/dcp_onlineExpert.json --rogue_agent_count $i --mode test --log_anime --best_epoch --test_general --log_time_trained 1582314635 --nGraphFilterTaps 2   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
    echo "\n" >> results.txt
    python main.py configs/dcp_onlineExpert.json --rogue_agent_count $i --mode test --log_anime --best_epoch --test_general --log_time_trained 1582034757  --nGraphFilterTaps 3   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
    echo "\n" >> results.txt
done