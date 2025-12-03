clc; close all;

%% 1. 데이터 로드
q_traj_path   = 'q_traj.csv';
q_dot_path    = 'q_dot_traj.csv';
body_path     = 'body_orientation.csv';
% 웨이포인트 파일 경로 (없으면 아래에서 임의 생성)
wp_path       = 'waypoints.csv'; 

% CSV 읽기
q_traj_tbl = readtable(q_traj_path);
q_dot_tbl  = readtable(q_dot_path);
body_tbl   = readtable(body_path);

% 데이터 변환
t = q_traj_tbl.t;              % 시간 [s]
q_mat = q_traj_tbl{:, 2:end};  % Joint angles
qd_mat = q_dot_tbl{:, 2:end};  % Joint velocities

yaw     = body_tbl.yaw;
pitch   = body_tbl.pitch;
roll    = body_tbl.roll;
yaw_tgt   = body_tbl.yaw_target;
pitch_tgt = body_tbl.pitch_target;
roll_tgt  = body_tbl.roll_target;

%% 1-1. 웨이포인트 데이터 준비 (파일이 있으면 읽고, 없으면 예시로 생성)
wp_tbl = table2array(readtable(wp_path));


%% 2. 그래프 그리기
figure('Position', [100 100 900 800]);

% --- 1) Joint Angles & Waypoints ---
subplot(3,1,1);
hold on;
% (1) 궤적 선 그리기
p_lines = plot(t, q_mat, 'LineWidth', 1.2); 

% (2) 웨이포인트 마커 그리기 (선 색상과 깔맞춤)
% 'HandleVisibility','off'를 하면 범례(Legend)에 마커가 중복으로 뜨지 않습니다.
for i = 1:6
    % 'Color', 'k' 추가
    plot(2.5, wp_tbl(1,i), 'o', 'Color', 'k', 'MarkerSize', 5, 'LineWidth', 1, 'HandleVisibility', 'off');
    plot(5, wp_tbl(1,6+i), 'o', 'Color', 'k', 'MarkerSize', 5, 'LineWidth', 1, 'HandleVisibility', 'off');
    plot(7.5, wp_tbl(1,12+i), 'o', 'Color', 'k', 'MarkerSize', 5, 'LineWidth', 1, 'HandleVisibility', 'off');
end

hold off;
title('Joint Angles (Cubic Spline)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Angle [rad]', 'FontSize', 12);
grid on;
xlim([min(t) max(t)]);

% 범례 설정 (내부 위치, 글자 작게, 2열 배치)
legend_str = arrayfun(@(i) sprintf('J%d', i), 1:size(q_mat,2), 'UniformOutput', false);
legend(legend_str, ...
    'Location', 'best', ...  % 그래프 내부 최적의 위치
    'FontSize', 9, ...       % 글자 크기 축소
    'NumColumns', 2, ...     % 가로 2줄로 배치하여 높이 절약
    'Box', 'on');            % 범례 박스 표시
set(gca, 'FontSize', 11);


% --- 2) Joint Velocities ---
subplot(3,1,2);
plot(t, qd_mat, 'LineWidth', 1.2);
title('Joint Velocities', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Vel [rad/s]', 'FontSize', 12);
grid on;
xlim([min(t) max(t)]);

% 속도 그래프는 범례가 너무 많으면 지저분하므로 생략하거나 간소화
% 필요하다면 위와 동일하게 설정
% legend(legend_str, 'Location', 'best', 'FontSize', 8, 'NumColumns', 2);
set(gca, 'FontSize', 11);


% --- 3) Body Orientation (Euler) ---
subplot(3,1,3);
hold on;
% 실제 궤적
plot(t, yaw,   'r-',  'LineWidth', 1.5);
plot(t, pitch, 'g-',  'LineWidth', 1.5);
plot(t, roll,  'b-',  'LineWidth', 1.5);
% 목표 궤적
plot(t, yaw_tgt,   'r--', 'LineWidth', 1.0);
plot(t, pitch_tgt, 'g--', 'LineWidth', 1.0);
plot(t, roll_tgt,  'b--', 'LineWidth', 1.0);
hold off;

title('Body Orientation (Euler)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Time [s]', 'FontSize', 12);
ylabel('Angle [rad]', 'FontSize', 12);
grid on;
xlim([min(t) max(t)]);

% 범례 설정
legend({'Yaw', 'Pitch', 'Roll', 'Yaw_{ref}', 'Pitch_{ref}', 'Roll_{ref}'}, ...
    'Location', 'best', ...
    'FontSize', 9, ...
    'NumColumns', 2); % 가로 3줄 배치
set(gca, 'FontSize', 11);