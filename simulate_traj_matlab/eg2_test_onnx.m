clear all;
rng('default');
filepath = fileparts(mfilename('fullpath'));
parts = strsplit(filepath, filesep);
parent_path = strjoin(parts(1:end-1), filesep);
result_dir = fullfile(parent_path, 'eg2_results', '001');
controller_dir = result_dir;
modelfile = fullfile(controller_dir, 'controller.onnx');

params = importONNXFunction(modelfile, "controller");

controller([0,0,0], params)
controller([1,1,1], params)
controller([2,2,2], params)
controller([3,3,3], params)
controller([-3,-3,-3], params)