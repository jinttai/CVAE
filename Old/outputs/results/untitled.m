clc; close all; clear;

waypoints_vec = table2array(readtable("waypoints.csv"));

waypoint = reshape(waypoints_vec,6,3)';

simout = sim('sim_SC_ur10e.slx');

time = simout.euler_angle_ZYX.Time;
quat = squeeze(simout.quaternion.Data)';
euler = simout.euler_angle_ZYX.Data;

data_to_save = [time, quat, euler];

writematrix(data_to_save, 'sim_result.csv');