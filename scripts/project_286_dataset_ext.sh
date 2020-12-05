cd ../
# No Online Expert

declare -a files=("configs/dcp_ECBS_LLQ1.json" "configs/dcp_ECBS_LLQ5.json" "configs/dcp_ECBS_edge1.json" "configs/dcp_ECBS_edge5.json")

for file in ${files[@]}; do
	echo "Config File for Experiment:" >> results.txt
	echo $file >> results.txt
	echo "\n" >> results.txt

	# K=1
	python main.py $file --mode test --log_anime --best_epoch --test_general --log_time_trained 1582029525  --nGraphFilterTaps 1   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20

	# K=2
	python main.py $file --mode test --log_anime --best_epoch --test_general --log_time_trained 1582028194  --nGraphFilterTaps 2   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20

	# K=3
	python main.py $file --mode test --log_anime --best_epoch --test_general --log_time_trained 1582028876  --nGraphFilterTaps 3   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20

	echo "\n" >> results.txt
done

# Online Expert

declare -a online_files=("configs/dcp_onlineExpert_LLQ1.json" "configs/dcp_onlineExpert_LLQ5.json" "configs/dcp_onlineExpert_edge1.json" "configs/dcp_onlineExpert_edge5.json")

for file in ${online_files[@]}; do
	echo "Config File for Experiment:" >> results.txt
	echo $file >> results.txt
	echo "\n" >> results.txt
	# K=2
	python main.py $file --mode test --log_anime --best_epoch --test_general --log_time_trained 1582314635 --nGraphFilterTaps 2   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
	# K=3
	python main.py $file --mode test --log_anime --best_epoch --test_general --log_time_trained 1582034757  --nGraphFilterTaps 3   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
	echo "\n" >> results.txt
done
