LOCAL_DIR='/Users/vtek/gnn_pathplanning/'
cd ../utils/
# No OE
# K = 1
#python visualize.py --nGraphFilterTaps 1 --type dcp --log_time_trained 1582029525 --caseId 00011 --local_dir $LOCAL_DIR
# K = 2
python visualize.py --nGraphFilterTaps 2 --type dcp --log_time_trained 1582028194 --caseId 00000 --local_dir $LOCAL_DIR
# K = 3
python visualize.py --nGraphFilterTaps 3 --type dcp --log_time_trained 1582028876 --caseId 00000 --local_dir $LOCAL_DIR

# OE
# K = 2
python visualize.py --nGraphFilterTaps 2 --type dcpOE --log_time_trained 1582314635 --caseId 00000 --local_dir $LOCAL_DIR
# K = 3
python visualize.py --nGraphFilterTaps 3 --type dcpOE --log_time_trained 1582034757 --caseId 00000 --local_dir $LOCAL_DIR