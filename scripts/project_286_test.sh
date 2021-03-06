cd ../
# No Online Expert
# K=1

files=("configs/dcp_ECBS.json")

for file in "$files"; do
	python main.py $file --mode test --log_anime --best_epoch --test_general --log_time_trained 1582029525  --nGraphFilterTaps 1   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20

	# K=2
	python main.py $file --mode test --log_anime --best_epoch --test_general --log_time_trained 1582028194  --nGraphFilterTaps 2   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20

	# K=3
	python main.py $file --mode test --log_anime --best_epoch --test_general --log_time_trained 1582028876  --nGraphFilterTaps 3   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
done

# Online Expert

online_files=("configs/dcp_onlineExpert.json")
# K=2

for file in "$online_files"; do
	python main.py $file --mode test --log_anime --best_epoch --test_general --log_time_trained 1582314635 --nGraphFilterTaps 2   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
	# K=3
	python main.py $file --mode test --log_anime --best_epoch --test_general --log_time_trained 1582034757  --nGraphFilterTaps 3   --map_w 20 --num_agents 10 --trained_num_agents 10 --trained_map_w 20
done