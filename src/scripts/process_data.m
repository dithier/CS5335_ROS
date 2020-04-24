clear all
close all
clc

%% extract data

path = '/Users/ithier/Documents/CS5335/Final_Project/CS5335_ROS/results/';
filename = strcat(path, 'combined_results.xls');
[num, txt, raw] = xlsread(filename);

all_our_position = num(:, 2:4);
all_our_orientation = num(:, 5:8);

all_true_position = num(:, 9:11);
all_true_orientation = num(:, 12:15);

% remove data where we don't have ground truth
no_truth_indices = find(all_true_orientation(:, 1) == 0);
good_indices = find(all_true_orientation(:, 1) ~= 0);

our_position = all_our_position(good_indices, :);
our_orientation = all_our_orientation(good_indices, :);
true_position = all_true_position(good_indices, :);
true_orientation = all_true_orientation(good_indices, :);

%% calculate statistics
position_errors = get_pos_error(our_position, true_position);
orientation_errors = get_ori_error(our_orientation, true_orientation);

avg_pos_error = mean(position_errors);
std_pos_error = std(position_errors);

avg_ori_error = mean(orientation_errors);
std_ori_error = std(orientation_errors);

%% make plots
n = size(our_position, 1);
frame = 1:n;

figure
bar(frame, position_errors)
title('Position Errors by Frame')
ylabel('Error (m)')
xlabel('Frame number')


figure
bar(frame, orientation_errors)
title('Orientation Errors by Frame')
ylabel('Error (radians of rotation)')
xlabel('Frame number')
%% Recalculate errors after removing outliers
ind = find(position_errors(:) < 3 & orientation_errors(:) < 1);

pos_errors = get_pos_error(our_position(ind, :), true_position(ind, :));
ori_errors = get_ori_error(our_orientation(ind, :), true_orientation(ind, :));

avg_pos_error = mean(pos_errors);
fprintf('avg position error %f\n', avg_pos_error);
std_pos_error = std(pos_errors);
fprintf('std position error %f\n', std_pos_error);

avg_ori_error = mean(ori_errors);
fprintf('avg ori error %f\n', avg_ori_error);
std_ori_error = std(ori_errors);
fprintf('std ori error %f\n', std_ori_error);

%% make plots without outliers
n = size(pos_errors, 2);
frame = 1:n;

figure
bar(frame, pos_errors)
title('Position Errors by Frame Without Outliers')
ylabel('Error (m)')
xlabel('Frame number')


figure
bar(frame, ori_errors)
title('Orientation Errors by Frame Without Outliers')
ylabel('Error (radians of rotation)')
xlabel('Frame number')

%% Helper functions
function error = get_pos_error(pos, truth)
    n = size(pos, 1);
    error = zeros(n, 1);
    for i = 1: n
        error(i, 1) = norm(pos(i, :) - truth(i, :));
    end
    error = error';
end

function error = get_ori_error(ori, truth)
    n = size(ori, 1);
    error = zeros(n, 1);
    for i = 1:n
        q1 = ori(i, :);
        q2 = truth(i, :);
        q = dot(q1, q2); 
        e = acos(2 * q^2 - 1);
        error(i, 1) = e;
    end
    error = error';
end 
