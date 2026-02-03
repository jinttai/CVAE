clc; close all; clear;

%% 1. 데이터 로드 (가상 데이터 생성 또는 파일 로드)
% 사용자의 파일이 없으므로, 코드가 작동하도록 임시 데이터를 생성합니다.
% 실제 사용 시에는 아래 'if true' 블록을 제거하고 파일 로드 부분만 남기시면 됩니다.

if exist('q_traj.csv', 'file')
    % 파일이 있을 경우 로드
    q_traj_tbl = readtable('q_traj.csv');
    q_dot_tbl  = readtable('q_dot_traj.csv');
    body_tbl   = readtable('body_orientation.csv');
    if exist('waypoints.csv', 'file')
        wp_tbl = table2array(readtable('waypoints.csv'));
    else
        wp_tbl = rand(1, 18); % 임시
    end
    
    t = q_traj_tbl.t;
    q_mat = q_traj_tbl{:, 2:end};
    qd_mat = q_dot_tbl{:, 2:end};
    
    yaw = body_tbl.yaw; pitch = body_tbl.pitch; roll = body_tbl.roll;
    yaw_tgt = body_tbl.yaw_target; pitch_tgt = body_tbl.pitch_target; roll_tgt = body_tbl.roll_target;
else
    % [테스트용 임시 데이터 생성]
    t = (0:0.01:10)';
    q_mat = sin(t + (1:6)); 
    qd_mat = cos(t + (1:6));
    yaw = sin(t); pitch = cos(t); roll = sin(2*t);
    yaw_tgt = sin(t); pitch_tgt = cos(t); roll_tgt = sin(2*t);
    wp_tbl = rand(1, 18); % 6 joints * 3 waypoints
end

%% 2. 스타일 정의 (논문용 포맷)
% 공통 스타일 변수
line_width_plot = 1.5;    % 데이터 선 두께
line_width_ref  = 1.2;    % 참조/목표값 선 두께
line_width_axis = 1.2;    % 축 선 두께
font_size_title = 14;     % 제목 폰트 크기
font_size_label = 12;     % 축 라벨 폰트 크기
font_size_tick  = 11;     % 눈금 폰트 크기
font_weight     = 'bold'; % 폰트 굵기

% 색상 팔레트 (Joint 6개에 대한 고유 색상 지정)
joint_colors = lines(size(q_mat, 2)); 

%% 3. 그래프 그리기
figure('Position', [100 100 900 800]);

% --- 1) Joint Angles & Waypoints ---
subplot(3,1,1);
hold on;

% (1) 궤적 선 그리기 (루프를 돌며 색상 지정)
for i = 1:size(q_mat, 2)
    plot(t, q_mat(:, i), 'Color', joint_colors(i,:), 'LineWidth', line_width_plot);
end

% (2) 웨이포인트 마커 그리기 (스타일 개선)
% MarkerFaceColor를 'w'(흰색)으로 하여 선 위에서도 잘 보이게 설정
wp_times = [2.5, 5, 7.5]; % 웨이포인트 시간 가정
for k = 1:3 % 웨이포인트 개수 (3개 시점)
    for i = 1:6 % Joint 개수
        wp_idx = (k-1)*6 + i;
        % 범례에 표시되지 않도록 HandleVisibility off
        plot(wp_times(k), wp_tbl(1, wp_idx), 'o', ...
            'Color', 'k', ...              % 테두리 검정
            'MarkerFaceColor', 'w', ...    % 내부 흰색 (가독성 향상)
            'MarkerSize', 6, ...
            'LineWidth', 1, ...
            'HandleVisibility', 'off');
    end
end
hold off;

% 축 및 라벨 설정
title('Joint Angles (Cubic Spline)', 'FontSize', font_size_title, 'FontWeight', font_weight);
ylabel('Angle [rad]', 'FontSize', font_size_label, 'FontWeight', font_weight);
grid on; box on;
xlim([min(t) max(t)]);
set(gca, 'LineWidth', line_width_axis, 'FontSize', font_size_tick, 'FontWeight', font_weight);

% 범례 설정
legend_str = arrayfun(@(i) sprintf('J%d', i), 1:size(q_mat,2), 'UniformOutput', false);
legend(legend_str, 'Location', 'eastoutside', 'FontSize', 9, 'Box', 'on');


% --- 2) Joint Velocities ---
subplot(3,1,2);
hold on;
% 위와 동일한 색상 매칭을 위해 루프 사용
for i = 1:size(qd_mat, 2)
    plot(t, qd_mat(:, i), 'Color', joint_colors(i,:), 'LineWidth', line_width_plot);
end
hold off;

title('Joint Velocities', 'FontSize', font_size_title, 'FontWeight', font_weight);
ylabel('Vel [rad/s]', 'FontSize', font_size_label, 'FontWeight', font_weight);
grid on; box on;
xlim([min(t) max(t)]);
set(gca, 'LineWidth', line_width_axis, 'FontSize', font_size_tick, 'FontWeight', font_weight);
% 속도 그래프는 복잡하므로 범례 생략 (필요시 주석 해제)
% legend(legend_str, 'Location', 'eastoutside', 'FontSize', 9);


% --- 3) Body Orientation (Euler) ---
subplot(3,1,3);
hold on;

% 색상 정의 (RGB) - 원색보다 살짝 톤 다운된 전문적인 색상
c_yaw   = [0.8500 0.3250 0.0980]; % Red계열 (Orange-Red)
c_pitch = [0.4660 0.6740 0.1880]; % Green계열
c_roll  = [0.0000 0.4470 0.7410]; % Blue계열

% 목표 궤적 (점선, 약간 투명하게 혹은 얇게) -> 범례 순서를 위해 먼저 그림
h_tgt(1) = plot(t, yaw_tgt,   '--', 'Color', c_yaw,   'LineWidth', line_width_ref);
h_tgt(2) = plot(t, pitch_tgt, '--', 'Color', c_pitch, 'LineWidth', line_width_ref);
h_tgt(3) = plot(t, roll_tgt,  '--', 'Color', c_roll,  'LineWidth', line_width_ref);

% 실제 궤적 (실선, 두껍게)
h_act(1) = plot(t, yaw,   '-',  'Color', c_yaw,   'LineWidth', line_width_plot);
h_act(2) = plot(t, pitch, '-',  'Color', c_pitch, 'LineWidth', line_width_plot);
h_act(3) = plot(t, roll,  '-',  'Color', c_roll,  'LineWidth', line_width_plot);

hold off;

title('Body Orientation (Euler)', 'FontSize', font_size_title, 'FontWeight', font_weight);
xlabel('Time [s]', 'FontSize', font_size_label, 'FontWeight', font_weight);
ylabel('Angle [rad]', 'FontSize', font_size_label, 'FontWeight', font_weight);
grid on; box on;
xlim([min(t) max(t)]);
set(gca, 'LineWidth', line_width_axis, 'FontSize', font_size_tick, 'FontWeight', font_weight);

% 범례 설정 (순서 재배치: Yaw끼리, Pitch끼리 묶어서 보기 좋게)
legend([h_act(1), h_tgt(1), h_act(2), h_tgt(2), h_act(3), h_tgt(3)], ...
       {'Yaw', 'Yaw_{ref}', 'Pitch', 'Pitch_{ref}', 'Roll', 'Roll_{ref}'}, ...
       'Location', 'eastoutside', 'FontSize', 9, 'NumColumns', 1);

% 전체 레이아웃 조정 (여백 최적화)
sgtitle('Trajectory Tracking Performance', 'FontSize', 16, 'FontWeight', 'bold');