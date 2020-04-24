clear all
close all
clc

%% Get data

% import bag
bag = rosbag('ar_data.bag');

% get messages from ar track alvar topic
b_select = select(bag, 'Topic', '/Ar_Pose');
pose_msgs = readMessages(b_select, 'DataFormat', 'struct');

% get positions
position_X = cellfun(@(m) m.Position.X, pose_msgs);
position_Y = cellfun(@(m) m.Position.Y, pose_msgs);
position_Z = cellfun(@(m) m.Position.Z, pose_msgs);
position = [position_X, position_Y, position_Z];

% get orientations
ori_X = cellfun(@(m) m.Orientation.X, pose_msgs);
ori_Y = cellfun(@(m) m.Orientation.Y, pose_msgs);
ori_Z = cellfun(@(m) m.Orientation.Z, pose_msgs);
ori_W = cellfun(@(m) m.Orientation.W, pose_msgs);
orientation = [ori_X, ori_Y, ori_Z, ori_W];

% put all data together
data = [position, orientation];

% Write data to file
path = '/Users/ithier/Documents/CS5335/Final_Project/CS5335_ROS/results/';
filename = strcat(path, 'ar_alvar_results.csv');
csvwrite(filename, data);

