clc; close all; clear;

%% 1. 데이터 정의 (실제 실험 데이터로 변경 필요)
% 예시: 연산 시간 (단위: 초) [Pure BFGS, MLP + BFGS, Proposed]
calc_time = [18.2, 5.2, 3.1]; 
% 예시: 표준 편차 (STD) 데이터 - 사용자 요청에 의해 추가됨 (실제 실험 데이터로 변경 필요)
calc_std = [1.5, 0.8, 0.3]; 

% x축 라벨 정의 (Adam 제거)
method_names = {'Pure BFGS', 'MLP + BFGS', 'Proposed'};
num_methods = length(calc_time);

%% 2. 그래프 그리기
figure('Position', [100, 100, 800, 600]);

% 막대 그래프 생성
bar_width = 0.6;
b = bar(calc_time, bar_width); 
hold on;

% 표준 편차 (Error Bar) 추가
% 'none' 라인 스타일로 막대 위에 수직 에러 바만 표시
x_center = 1:num_methods;
e = errorbar(x_center, calc_time, calc_std, 'k', 'LineWidth', 1.5, 'CapSize', 18);
e.LineStyle = 'none'; % 에러 바를 잇는 선 제거
hold off;

%% 3. 스타일 설정 (논문용 포맷팅)
% 3-1. 색상 설정: Proposed만 강조하고 나머지는 회색 처리
b.FaceColor = 'flat'; % 개별 색상 지정 활성화
base_color = [0.7, 0.7, 0.7]; % 회색 (Baselines)
prop_color = [0, 0.4470, 0.7410]; % 파란색 or 강조색 (Proposed)

% 색상 설정 (3개 데이터 기준): 1~2번째는 회색, 3번째(Proposed)는 강조색
b.CData(1,:) = base_color;
b.CData(2,:) = base_color;
b.CData(3,:) = prop_color;

% 3-2. 축 및 라벨 설정
set(gca, 'XTickLabel', method_names, 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Computation Time [s]', 'FontSize', 14, 'FontWeight', 'bold');
title('Comparison of Calculation Time', 'FontSize', 16, 'FontWeight', 'bold');
grid on;
box on;       % 그래프 테두리 닫기
set(gca, 'LineWidth', 1.2); % 축 선 두께

% y축 범위 (데이터 최대값 + STD 최대값 보다 20% 높게 잡아서 여유 공간 확보)
max_val = max(calc_time + calc_std);
ylim([0, max_val * 1.2]); 

%% 4. 막대 위에 수치 텍스트 표시
% 각 막대 높이 바로 위에 텍스트 추가 (평균값 + STD 표시)
for i = 1:num_methods
    % 텍스트 위치: 평균값 + STD 만큼 위
    text_y_pos = calc_time(i) + calc_std(i);
    % 텍스트 내용: 평균값 $\pm$ 표준 편차 형태로 포맷팅
    % $\pm$ 기호를 위해 LaTeX 포맷을 사용합니다.
    text_str = sprintf('%.2f $\\pm$ %.2f s', calc_time(i), calc_std(i));
    
    text(i, text_y_pos, text_str, ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', ...
        'FontSize', 11, 'FontWeight', 'bold', ...
        'Interpreter', 'latex'); % LaTeX 인터프리터 설정
end