 THE Configuration of your experiment ..
{'Start_onlineExpert': 20,
 'Use_infoMode': 0,
 'agent': 'DecentralPlannerAgentLocal',
 'async_loading': True,
 'batch_size': 64,
 'best_epoch': True,
 'checkpoint_file': 'checkpoint.pth.tar',
 'commR': 6,
 'comm_dropout_param': None,
 'con_train': False,
 'cuda': True,
 'data_loader': 'DecentralPlannerDataLoader',
 'data_loader_workers': 4,
 'data_root': '/Users/shuvomsadhuka/Desktop/Junior/Fall/CS286/project/gnn_pathplanning_new',
 'exp_net': 'dcp',
 'feature_noise_std': None,
 'gpu_device': 0,
 'hiddenFeatures': 0,
 'id_map': 0,
 'lastest_epoch': False,
 'learning_rate': 0.001,
 'log_anime': True,
 'log_interval': 500,
 'map_density': 1,
 'map_h': 20,
 'map_noise_prob': None,
 'map_shift_units': None,
 'map_type': 'map',
 'map_w': 20,
 'max_epoch': 150,
 'mode': 'test',
 'momentum': 0.5,
 'move_noise_std': None,
 'nGraphFilterTaps': 1,
 'num_agents': 10,
 'num_test_trainingSet': 500,
 'num_testset': 10,
 'num_validset': 200,
 'penaltyCollision': 0.05,
 'pin_memory': True,
 'rate_maxstep': 2,
 'rogue_agent_count': None,
 'save_data': '/Users/shuvomsadhuka/Desktop/Junior/Fall/CS286/project/gnn_pathplanning_new',
 'save_tb_data': '/Users/shuvomsadhuka/Desktop/Junior/Fall/CS286/project/gnn_pathplanning_new',
 'seed': 1337,
 'sybil_attack_count': None,
 'test_batch_size': 1,
 'test_epoch': 0,
 'test_general': True,
 'train_TL': False,
 'trained_map_density': 1,
 'trained_map_h': 20,
 'trained_map_type': 'map',
 'trained_map_w': 20,
 'trained_num_agents': 10,
 'valid_batch_size': 1,
 'validate_every': 4,
 'weight_decay': 1e-05}
 *************************************** 
The experiment name is dcp
 *************************************** 
BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB

Input size into first GNN: 128

Output size of first GNN: 128

FILTER TAPS [1]

DecentralPlannerNet(
  (ConvLayers): Sequential(
    (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU(inplace=True)
    (7): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (9): ReLU(inplace=True)
    (10): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (11): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (12): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (13): ReLU(inplace=True)
    (14): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (16): ReLU(inplace=True)
    (17): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (compressMLP): Sequential(
    (0): Linear(in_features=128, out_features=128, bias=True)
    (1): ReLU(inplace=True)
  )
  (GFL): Sequential(
    (0): GraphFilterBatch(in_features=128, out_features=128, filter_taps=1, edge_features=1, bias=True, no GSO stored)
    (1): ReLU(inplace=True)
  )
  (actionsMLP): Sequential(
    (0): Linear(in_features=128, out_features=5, bias=True)
  )
)
1e-05
run on multirobotsim with collision shielding
-------test------------
Experiment on dcp_map20x20_rho1_10Agent/K1_HS0/1582029525 finished.
Please wait while finalizing the operation.. Thank you
################## End of testing ################## 
Computation time:42.894453

 THE Configuration of your experiment ..
{'Start_onlineExpert': 20,
 'Use_infoMode': 0,
 'agent': 'DecentralPlannerAgentLocal',
 'async_loading': True,
 'batch_size': 64,
 'best_epoch': True,
 'checkpoint_file': 'checkpoint.pth.tar',
 'commR': 6,
 'comm_dropout_param': None,
 'con_train': False,
 'cuda': True,
 'data_loader': 'DecentralPlannerDataLoader',
 'data_loader_workers': 4,
 'data_root': '/Users/shuvomsadhuka/Desktop/Junior/Fall/CS286/project/gnn_pathplanning_new',
 'exp_net': 'dcp',
 'feature_noise_std': None,
 'gpu_device': 0,
 'hiddenFeatures': 0,
 'id_map': 0,
 'lastest_epoch': False,
 'learning_rate': 0.001,
 'log_anime': True,
 'log_interval': 500,
 'map_density': 1,
 'map_h': 20,
 'map_noise_prob': None,
 'map_shift_units': None,
 'map_type': 'map',
 'map_w': 20,
 'max_epoch': 150,
 'mode': 'test',
 'momentum': 0.5,
 'move_noise_std': None,
 'nGraphFilterTaps': 2,
 'num_agents': 10,
 'num_test_trainingSet': 500,
 'num_testset': 10,
 'num_validset': 200,
 'penaltyCollision': 0.05,
 'pin_memory': True,
 'rate_maxstep': 2,
 'rogue_agent_count': None,
 'save_data': '/Users/shuvomsadhuka/Desktop/Junior/Fall/CS286/project/gnn_pathplanning_new',
 'save_tb_data': '/Users/shuvomsadhuka/Desktop/Junior/Fall/CS286/project/gnn_pathplanning_new',
 'seed': 1337,
 'sybil_attack_count': None,
 'test_batch_size': 1,
 'test_epoch': 0,
 'test_general': True,
 'train_TL': False,
 'trained_map_density': 1,
 'trained_map_h': 20,
 'trained_map_type': 'map',
 'trained_map_w': 20,
 'trained_num_agents': 10,
 'valid_batch_size': 1,
 'validate_every': 4,
 'weight_decay': 1e-05}
 *************************************** 
The experiment name is dcp
 *************************************** 
BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB

Input size into first GNN: 128

Output size of first GNN: 128

FILTER TAPS [2]

DecentralPlannerNet(
  (ConvLayers): Sequential(
    (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU(inplace=True)
    (7): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (9): ReLU(inplace=True)
    (10): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (11): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (12): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (13): ReLU(inplace=True)
    (14): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (16): ReLU(inplace=True)
    (17): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (compressMLP): Sequential(
    (0): Linear(in_features=128, out_features=128, bias=True)
    (1): ReLU(inplace=True)
  )
  (GFL): Sequential(
    (0): GraphFilterBatch(in_features=128, out_features=128, filter_taps=2, edge_features=1, bias=True, no GSO stored)
    (1): ReLU(inplace=True)
  )
  (actionsMLP): Sequential(
    (0): Linear(in_features=128, out_features=5, bias=True)
  )
)
1e-05
run on multirobotsim with collision shielding
-------test------------
Experiment on dcp_map20x20_rho1_10Agent/K2_HS0/1582028194 finished.
Please wait while finalizing the operation.. Thank you
################## End of testing ################## 
Computation time:26.303962000000002

 THE Configuration of your experiment ..
{'Start_onlineExpert': 20,
 'Use_infoMode': 0,
 'agent': 'DecentralPlannerAgentLocal',
 'async_loading': True,
 'batch_size': 64,
 'best_epoch': True,
 'checkpoint_file': 'checkpoint.pth.tar',
 'commR': 6,
 'comm_dropout_param': None,
 'con_train': False,
 'cuda': True,
 'data_loader': 'DecentralPlannerDataLoader',
 'data_loader_workers': 4,
 'data_root': '/Users/shuvomsadhuka/Desktop/Junior/Fall/CS286/project/gnn_pathplanning_new',
 'exp_net': 'dcp',
 'feature_noise_std': None,
 'gpu_device': 0,
 'hiddenFeatures': 0,
 'id_map': 0,
 'lastest_epoch': False,
 'learning_rate': 0.001,
 'log_anime': True,
 'log_interval': 500,
 'map_density': 1,
 'map_h': 20,
 'map_noise_prob': None,
 'map_shift_units': None,
 'map_type': 'map',
 'map_w': 20,
 'max_epoch': 150,
 'mode': 'test',
 'momentum': 0.5,
 'move_noise_std': None,
 'nGraphFilterTaps': 3,
 'num_agents': 10,
 'num_test_trainingSet': 500,
 'num_testset': 10,
 'num_validset': 200,
 'penaltyCollision': 0.05,
 'pin_memory': True,
 'rate_maxstep': 2,
 'rogue_agent_count': None,
 'save_data': '/Users/shuvomsadhuka/Desktop/Junior/Fall/CS286/project/gnn_pathplanning_new',
 'save_tb_data': '/Users/shuvomsadhuka/Desktop/Junior/Fall/CS286/project/gnn_pathplanning_new',
 'seed': 1337,
 'sybil_attack_count': None,
 'test_batch_size': 1,
 'test_epoch': 0,
 'test_general': True,
 'train_TL': False,
 'trained_map_density': 1,
 'trained_map_h': 20,
 'trained_map_type': 'map',
 'trained_map_w': 20,
 'trained_num_agents': 10,
 'valid_batch_size': 1,
 'validate_every': 4,
 'weight_decay': 1e-05}
 *************************************** 
The experiment name is dcp
 *************************************** 
BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB

Input size into first GNN: 128

Output size of first GNN: 128

FILTER TAPS [3]

DecentralPlannerNet(
  (ConvLayers): Sequential(
    (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU(inplace=True)
    (7): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (9): ReLU(inplace=True)
    (10): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (11): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (12): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (13): ReLU(inplace=True)
    (14): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (16): ReLU(inplace=True)
    (17): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (compressMLP): Sequential(
    (0): Linear(in_features=128, out_features=128, bias=True)
    (1): ReLU(inplace=True)
  )
  (GFL): Sequential(
    (0): GraphFilterBatch(in_features=128, out_features=128, filter_taps=3, edge_features=1, bias=True, no GSO stored)
    (1): ReLU(inplace=True)
  )
  (actionsMLP): Sequential(
    (0): Linear(in_features=128, out_features=5, bias=True)
  )
)
1e-05
run on multirobotsim with collision shielding
-------test------------
Experiment on dcp_map20x20_rho1_10Agent/K3_HS0/1582028876 finished.
Please wait while finalizing the operation.. Thank you
################## End of testing ################## 
Computation time:29.778401000000002

 THE Configuration of your experiment ..
{'Start_onlineExpert': 0,
 'Use_infoMode': 0,
 'agent': 'DecentralPlannerAgentLocalWithOnlineExpert',
 'async_loading': True,
 'batch_size': 64,
 'best_epoch': True,
 'checkpoint_file': 'checkpoint.pth.tar',
 'commR': 6,
 'comm_dropout_param': None,
 'con_train': False,
 'cuda': False,
 'data_loader': 'DecentralPlannerDataLoader',
 'data_loader_workers': 4,
 'data_root': '/Users/shuvomsadhuka/Desktop/Junior/Fall/CS286/project/gnn_pathplanning_new',
 'exp_net': 'dcpOE',
 'feature_noise_std': None,
 'gpu_device': 0,
 'hiddenFeatures': 0,
 'id_map': 0,
 'lastest_epoch': False,
 'learning_rate': 0.001,
 'log_anime': True,
 'log_interval': 500,
 'map_density': 1,
 'map_h': 20,
 'map_noise_prob': None,
 'map_shift_units': None,
 'map_type': 'map',
 'map_w': 20,
 'max_epoch': 150,
 'mode': 'test',
 'momentum': 0.5,
 'move_noise_std': None,
 'nGraphFilterTaps': 2,
 'num_agents': 10,
 'num_test_trainingSet': 500,
 'num_testset': 10,
 'num_validset': 200,
 'penaltyCollision': 0.05,
 'pin_memory': True,
 'rate_maxstep': 2,
 'rogue_agent_count': None,
 'save_data': '/Users/shuvomsadhuka/Desktop/Junior/Fall/CS286/project/gnn_pathplanning_new',
 'save_tb_data': '/Users/shuvomsadhuka/Desktop/Junior/Fall/CS286/project/gnn_pathplanning_new',
 'seed': 1337,
 'sybil_attack_count': None,
 'test_batch_size': 1,
 'test_epoch': 0,
 'test_general': True,
 'train_TL': False,
 'trained_map_density': 1,
 'trained_map_h': 20,
 'trained_map_type': 'map',
 'trained_map_w': 20,
 'trained_num_agents': 10,
 'valid_batch_size': 1,
 'validate_every': 4,
 'weight_decay': 1e-05}
 *************************************** 
The experiment name is dcpOE
 *************************************** 
BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB

Input size into first GNN: 128

Output size of first GNN: 128

FILTER TAPS [2]

DecentralPlannerNet(
  (ConvLayers): Sequential(
    (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU(inplace=True)
    (7): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (9): ReLU(inplace=True)
    (10): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (11): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (12): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (13): ReLU(inplace=True)
    (14): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (16): ReLU(inplace=True)
    (17): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (compressMLP): Sequential(
    (0): Linear(in_features=128, out_features=128, bias=True)
    (1): ReLU(inplace=True)
  )
  (GFL): Sequential(
    (0): GraphFilterBatch(in_features=128, out_features=128, filter_taps=2, edge_features=1, bias=True, no GSO stored)
    (1): ReLU(inplace=True)
  )
  (actionsMLP): Sequential(
    (0): Linear(in_features=128, out_features=5, bias=True)
  )
)
1e-05
run on multirobotsim with collision shielding
-------test------------
Experiment on dcpOE_map20x20_rho1_10Agent/K2_HS0/1582314635 finished.
Please wait while finalizing the operation.. Thank you
################## End of testing ################## 
Computation time:23.382271

 THE Configuration of your experiment ..
{'Start_onlineExpert': 0,
 'Use_infoMode': 0,
 'agent': 'DecentralPlannerAgentLocalWithOnlineExpert',
 'async_loading': True,
 'batch_size': 64,
 'best_epoch': True,
 'checkpoint_file': 'checkpoint.pth.tar',
 'commR': 6,
 'comm_dropout_param': None,
 'con_train': False,
 'cuda': False,
 'data_loader': 'DecentralPlannerDataLoader',
 'data_loader_workers': 4,
 'data_root': '/Users/shuvomsadhuka/Desktop/Junior/Fall/CS286/project/gnn_pathplanning_new',
 'exp_net': 'dcpOE',
 'feature_noise_std': None,
 'gpu_device': 0,
 'hiddenFeatures': 0,
 'id_map': 0,
 'lastest_epoch': False,
 'learning_rate': 0.001,
 'log_anime': True,
 'log_interval': 500,
 'map_density': 1,
 'map_h': 20,
 'map_noise_prob': None,
 'map_shift_units': None,
 'map_type': 'map',
 'map_w': 20,
 'max_epoch': 150,
 'mode': 'test',
 'momentum': 0.5,
 'move_noise_std': None,
 'nGraphFilterTaps': 3,
 'num_agents': 10,
 'num_test_trainingSet': 500,
 'num_testset': 10,
 'num_validset': 200,
 'penaltyCollision': 0.05,
 'pin_memory': True,
 'rate_maxstep': 2,
 'rogue_agent_count': None,
 'save_data': '/Users/shuvomsadhuka/Desktop/Junior/Fall/CS286/project/gnn_pathplanning_new',
 'save_tb_data': '/Users/shuvomsadhuka/Desktop/Junior/Fall/CS286/project/gnn_pathplanning_new',
 'seed': 1337,
 'sybil_attack_count': None,
 'test_batch_size': 1,
 'test_epoch': 0,
 'test_general': True,
 'train_TL': False,
 'trained_map_density': 1,
 'trained_map_h': 20,
 'trained_map_type': 'map',
 'trained_map_w': 20,
 'trained_num_agents': 10,
 'valid_batch_size': 1,
 'validate_every': 4,
 'weight_decay': 1e-05}
 *************************************** 
The experiment name is dcpOE
 *************************************** 
BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB

Input size into first GNN: 128

Output size of first GNN: 128

FILTER TAPS [3]

DecentralPlannerNet(
  (ConvLayers): Sequential(
    (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU(inplace=True)
    (7): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (9): ReLU(inplace=True)
    (10): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (11): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (12): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (13): ReLU(inplace=True)
    (14): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (16): ReLU(inplace=True)
    (17): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (compressMLP): Sequential(
    (0): Linear(in_features=128, out_features=128, bias=True)
    (1): ReLU(inplace=True)
  )
  (GFL): Sequential(
    (0): GraphFilterBatch(in_features=128, out_features=128, filter_taps=3, edge_features=1, bias=True, no GSO stored)
    (1): ReLU(inplace=True)
  )
  (actionsMLP): Sequential(
    (0): Linear(in_features=128, out_features=5, bias=True)
  )
)
1e-05
run on multirobotsim with collision shielding
-------test------------
Experiment on dcpOE_map20x20_rho1_10Agent/K3_HS0/1582034757 finished.
Please wait while finalizing the operation.. Thank you
################## End of testing ################## 
Computation time:29.223522000000003

 THE Configuration of your experiment ..
{'Start_onlineExpert': 20,
 'Use_infoMode': 0,
 'agent': 'DecentralPlannerAgentLocal',
 'async_loading': True,
 'batch_size': 64,
 'best_epoch': True,
 'checkpoint_file': 'checkpoint.pth.tar',
 'commR': 6,
 'comm_dropout_param': None,
 'con_train': False,
 'cuda': True,
 'data_loader': 'DecentralPlannerDataLoader',
 'data_loader_workers': 4,
 'data_root': '/Users/shuvomsadhuka/Desktop/Junior/Fall/CS286/project/gnn_pathplanning_new',
 'exp_net': 'dcp',
 'feature_noise_std': None,
 'gpu_device': 0,
 'hiddenFeatures': 0,
 'id_map': 0,
 'lastest_epoch': False,
 'learning_rate': 0.001,
 'log_anime': True,
 'log_interval': 500,
 'map_density': 1,
 'map_h': 20,
 'map_noise_prob': None,
 'map_shift_units': None,
 'map_type': 'map',
 'map_w': 20,
 'max_epoch': 150,
 'mode': 'test',
 'momentum': 0.5,
 'move_noise_std': None,
 'nGraphFilterTaps': 1,
 'num_agents': 10,
 'num_test_trainingSet': 500,
 'num_testset': 100,
 'num_validset': 200,
 'penaltyCollision': 0.05,
 'pin_memory': True,
 'rate_maxstep': 2,
 'rogue_agent_count': None,
 'save_data': '/Users/shuvomsadhuka/Desktop/Junior/Fall/CS286/project/gnn_pathplanning_new',
 'save_tb_data': '/Users/shuvomsadhuka/Desktop/Junior/Fall/CS286/project/gnn_pathplanning_new',
 'seed': 1337,
 'sybil_attack_count': None,
 'test_batch_size': 1,
 'test_epoch': 0,
 'test_general': True,
 'train_TL': False,
 'trained_map_density': 1,
 'trained_map_h': 20,
 'trained_map_type': 'map',
 'trained_map_w': 20,
 'trained_num_agents': 10,
 'valid_batch_size': 1,
 'validate_every': 4,
 'weight_decay': 1e-05}
 *************************************** 
The experiment name is dcp
 *************************************** 
BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB

Input size into first GNN: 128

Output size of first GNN: 128

FILTER TAPS [1]

DecentralPlannerNet(
  (ConvLayers): Sequential(
    (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU(inplace=True)
    (7): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (9): ReLU(inplace=True)
    (10): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (11): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (12): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (13): ReLU(inplace=True)
    (14): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (16): ReLU(inplace=True)
    (17): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (compressMLP): Sequential(
    (0): Linear(in_features=128, out_features=128, bias=True)
    (1): ReLU(inplace=True)
  )
  (GFL): Sequential(
    (0): GraphFilterBatch(in_features=128, out_features=128, filter_taps=1, edge_features=1, bias=True, no GSO stored)
    (1): ReLU(inplace=True)
  )
  (actionsMLP): Sequential(
    (0): Linear(in_features=128, out_features=5, bias=True)
  )
)
1e-05
run on multirobotsim with collision shielding
-------test------------
 THE Configuration of your experiment ..
{'Start_onlineExpert': 20,
 'Use_infoMode': 0,
 'agent': 'DecentralPlannerAgentLocal',
 'async_loading': True,
 'batch_size': 64,
 'best_epoch': True,
 'checkpoint_file': 'checkpoint.pth.tar',
 'commR': 6,
 'comm_dropout_param': None,
 'con_train': False,
 'cuda': True,
 'data_loader': 'DecentralPlannerDataLoader',
 'data_loader_workers': 4,
 'data_root': '/Users/shuvomsadhuka/Desktop/Junior/Fall/CS286/project/gnn_pathplanning_new',
 'exp_net': 'dcp',
 'feature_noise_std': None,
 'gpu_device': 0,
 'hiddenFeatures': 0,
 'id_map': 0,
 'lastest_epoch': False,
 'learning_rate': 0.001,
 'log_anime': True,
 'log_interval': 500,
 'map_density': 1,
 'map_h': 20,
 'map_noise_prob': None,
 'map_shift_units': None,
 'map_type': 'map',
 'map_w': 20,
 'max_epoch': 150,
 'mode': 'test',
 'momentum': 0.5,
 'move_noise_std': None,
 'nGraphFilterTaps': 2,
 'num_agents': 10,
 'num_test_trainingSet': 500,
 'num_testset': 100,
 'num_validset': 200,
 'penaltyCollision': 0.05,
 'pin_memory': True,
 'rate_maxstep': 2,
 'rogue_agent_count': None,
 'save_data': '/Users/shuvomsadhuka/Desktop/Junior/Fall/CS286/project/gnn_pathplanning_new',
 'save_tb_data': '/Users/shuvomsadhuka/Desktop/Junior/Fall/CS286/project/gnn_pathplanning_new',
 'seed': 1337,
 'sybil_attack_count': None,
 'test_batch_size': 1,
 'test_epoch': 0,
 'test_general': True,
 'train_TL': False,
 'trained_map_density': 1,
 'trained_map_h': 20,
 'trained_map_type': 'map',
 'trained_map_w': 20,
 'trained_num_agents': 10,
 'valid_batch_size': 1,
 'validate_every': 4,
 'weight_decay': 1e-05}
 *************************************** 
The experiment name is dcp
 *************************************** 
BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB

Input size into first GNN: 128

Output size of first GNN: 128

FILTER TAPS [2]

DecentralPlannerNet(
  (ConvLayers): Sequential(
    (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU(inplace=True)
    (7): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (9): ReLU(inplace=True)
    (10): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (11): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (12): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (13): ReLU(inplace=True)
    (14): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (16): ReLU(inplace=True)
    (17): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (compressMLP): Sequential(
    (0): Linear(in_features=128, out_features=128, bias=True)
    (1): ReLU(inplace=True)
  )
  (GFL): Sequential(
    (0): GraphFilterBatch(in_features=128, out_features=128, filter_taps=2, edge_features=1, bias=True, no GSO stored)
    (1): ReLU(inplace=True)
  )
  (actionsMLP): Sequential(
    (0): Linear(in_features=128, out_features=5, bias=True)
  )
)
1e-05
run on multirobotsim with collision shielding
-------test------------
 THE Configuration of your experiment ..
{'Start_onlineExpert': 20,
 'Use_infoMode': 0,
 'agent': 'DecentralPlannerAgentLocal',
 'async_loading': True,
 'batch_size': 64,
 'best_epoch': True,
 'checkpoint_file': 'checkpoint.pth.tar',
 'commR': 6,
 'comm_dropout_param': None,
 'con_train': False,
 'cuda': True,
 'data_loader': 'DecentralPlannerDataLoader',
 'data_loader_workers': 4,
 'data_root': '/Users/shuvomsadhuka/Desktop/Junior/Fall/CS286/project/gnn_pathplanning_new',
 'exp_net': 'dcp',
 'feature_noise_std': None,
 'gpu_device': 0,
 'hiddenFeatures': 0,
 'id_map': 0,
 'lastest_epoch': False,
 'learning_rate': 0.001,
 'log_anime': True,
 'log_interval': 500,
 'map_density': 1,
 'map_h': 20,
 'map_noise_prob': None,
 'map_shift_units': None,
 'map_type': 'map',
 'map_w': 20,
 'max_epoch': 150,
 'mode': 'test',
 'momentum': 0.5,
 'move_noise_std': None,
 'nGraphFilterTaps': 3,
 'num_agents': 10,
 'num_test_trainingSet': 500,
 'num_testset': 100,
 'num_validset': 200,
 'penaltyCollision': 0.05,
 'pin_memory': True,
 'rate_maxstep': 2,
 'rogue_agent_count': None,
 'save_data': '/Users/shuvomsadhuka/Desktop/Junior/Fall/CS286/project/gnn_pathplanning_new',
 'save_tb_data': '/Users/shuvomsadhuka/Desktop/Junior/Fall/CS286/project/gnn_pathplanning_new',
 'seed': 1337,
 'sybil_attack_count': None,
 'test_batch_size': 1,
 'test_epoch': 0,
 'test_general': True,
 'train_TL': False,
 'trained_map_density': 1,
 'trained_map_h': 20,
 'trained_map_type': 'map',
 'trained_map_w': 20,
 'trained_num_agents': 10,
 'valid_batch_size': 1,
 'validate_every': 4,
 'weight_decay': 1e-05}
 *************************************** 
The experiment name is dcp
 *************************************** 
BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB

Input size into first GNN: 128

Output size of first GNN: 128

FILTER TAPS [3]

DecentralPlannerNet(
  (ConvLayers): Sequential(
    (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU(inplace=True)
    (7): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (9): ReLU(inplace=True)
    (10): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (11): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (12): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (13): ReLU(inplace=True)
    (14): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (16): ReLU(inplace=True)
    (17): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (compressMLP): Sequential(
    (0): Linear(in_features=128, out_features=128, bias=True)
    (1): ReLU(inplace=True)
  )
  (GFL): Sequential(
    (0): GraphFilterBatch(in_features=128, out_features=128, filter_taps=3, edge_features=1, bias=True, no GSO stored)
    (1): ReLU(inplace=True)
  )
  (actionsMLP): Sequential(
    (0): Linear(in_features=128, out_features=5, bias=True)
  )
)
1e-05
run on multirobotsim with collision shielding
-------test------------
 THE Configuration of your experiment ..
{'Start_onlineExpert': 0,
 'Use_infoMode': 0,
 'agent': 'DecentralPlannerAgentLocalWithOnlineExpert',
 'async_loading': True,
 'batch_size': 64,
 'best_epoch': True,
 'checkpoint_file': 'checkpoint.pth.tar',
 'commR': 6,
 'comm_dropout_param': None,
 'con_train': False,
 'cuda': False,
 'data_loader': 'DecentralPlannerDataLoader',
 'data_loader_workers': 4,
 'data_root': '/Users/shuvomsadhuka/Desktop/Junior/Fall/CS286/project/gnn_pathplanning_new',
 'exp_net': 'dcpOE',
 'feature_noise_std': None,
 'gpu_device': 0,
 'hiddenFeatures': 0,
 'id_map': 0,
 'lastest_epoch': False,
 'learning_rate': 0.001,
 'log_anime': True,
 'log_interval': 500,
 'map_density': 1,
 'map_h': 20,
 'map_noise_prob': None,
 'map_shift_units': None,
 'map_type': 'map',
 'map_w': 20,
 'max_epoch': 150,
 'mode': 'test',
 'momentum': 0.5,
 'move_noise_std': None,
 'nGraphFilterTaps': 2,
 'num_agents': 10,
 'num_test_trainingSet': 500,
 'num_testset': 100,
 'num_validset': 200,
 'penaltyCollision': 0.05,
 'pin_memory': True,
 'rate_maxstep': 2,
 'rogue_agent_count': None,
 'save_data': '/Users/shuvomsadhuka/Desktop/Junior/Fall/CS286/project/gnn_pathplanning_new',
 'save_tb_data': '/Users/shuvomsadhuka/Desktop/Junior/Fall/CS286/project/gnn_pathplanning_new',
 'seed': 1337,
 'sybil_attack_count': None,
 'test_batch_size': 1,
 'test_epoch': 0,
 'test_general': True,
 'train_TL': False,
 'trained_map_density': 1,
 'trained_map_h': 20,
 'trained_map_type': 'map',
 'trained_map_w': 20,
 'trained_num_agents': 10,
 'valid_batch_size': 1,
 'validate_every': 4,
 'weight_decay': 1e-05}
 *************************************** 
The experiment name is dcpOE
 *************************************** 
BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB

Input size into first GNN: 128

Output size of first GNN: 128

FILTER TAPS [2]

DecentralPlannerNet(
  (ConvLayers): Sequential(
    (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU(inplace=True)
    (7): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (9): ReLU(inplace=True)
    (10): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (11): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (12): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (13): ReLU(inplace=True)
    (14): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (16): ReLU(inplace=True)
    (17): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (compressMLP): Sequential(
    (0): Linear(in_features=128, out_features=128, bias=True)
    (1): ReLU(inplace=True)
  )
  (GFL): Sequential(
    (0): GraphFilterBatch(in_features=128, out_features=128, filter_taps=2, edge_features=1, bias=True, no GSO stored)
    (1): ReLU(inplace=True)
  )
  (actionsMLP): Sequential(
    (0): Linear(in_features=128, out_features=5, bias=True)
  )
)
1e-05
run on multirobotsim with collision shielding
-------test------------
Experiment on dcpOE_map20x20_rho1_10Agent/K2_HS0/1582314635 finished.
Please wait while finalizing the operation.. Thank you
################## End of testing ################## 
Computation time:None

