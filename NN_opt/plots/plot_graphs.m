clc; close all; clear;

%% 1. 데이터 로드 (파일이 없으면 가상 데이터 생성)
if exist('v3.csv', 'file')
    cvae_training_data = readtable("v3.csv");
    cvae_epoch = cvae_training_data.epoch;
    cvae_loss = cvae_training_data.train_loss;
else
    % [테스트용 가상 데이터] 15000 Epoch 가정
    fprintf('데이터 파일이 없어 가상 데이터를 생성합니다.\n');
    cvae_epoch = (1:15000)';
    % 초기값 2.5에서 시작해 0.15로 수렴 + 노이즈 추가
    cvae_loss = 2.35 * exp(-cvae_epoch/1000) + 0.15 + 0.05 * randn(size(cvae_epoch));
end

%% 2. 스타일 정의 (논문용 포맷)
line_width_plot = 1.2;    % 데이터 선 두께
line_width_avg  = 2.0;    % 평균선 두께
line_width_axis = 1.2;    % 축 선 두께
font_size_title = 16;     % 제목 폰트 크기
font_size_label = 14;     % 축 라벨 폰트 크기
font_size_tick  = 12;     % 눈금 폰트 크기
font_weight     = 'bold'; % 폰트 굵기

% 색상 팔레트
main_color = [0, 0.4470, 0.7410]; % 표준 Blue
avg_color  = [0.8500, 0.3250, 0.0980]; % 강조용 Orange-Red

%% 3. 평균 계산 (수렴 구간)
% 데이터가 충분하지 않을 경우를 대비해 인덱스 안전장치 마련
idx_start = 1950;
if length(cvae_loss) < idx_start
    idx_start = max(1, floor(length(cvae_loss) * 0.9)); % 데이터가 짧으면 하위 10%만 사용
end

% 마지막 구간 평균 계산
average_loss = mean(cvae_loss(idx_start:end));

%% 4. 그래프 그리기
figure('Position', [100, 100, 800, 600]);
hold on;

% (1) 전체 Loss Plot (약간 투명도를 주거나 얇게 그려 노이즈가 많아도 보기에 부담 없게)
plot(cvae_epoch, cvae_loss, 'Color', main_color, 'LineWidth', 0.5, 'DisplayName', 'Training Loss');


hold off;

% 축 및 라벨 설정
title("Generative Model Training Result", 'FontSize', font_size_title, 'FontWeight', font_weight);
xlabel("Training Epoch", 'FontSize', font_size_label, 'FontWeight', font_weight);
ylabel("Loss", 'FontSize', font_size_label, 'FontWeight', font_weight);

% 그리드 및 박스 설정
grid on; 
box on;
set(gca, 'LineWidth', line_width_axis, 'FontSize', font_size_tick, 'FontWeight', font_weight);
xlim([min(cvae_epoch), max(cvae_epoch)]);

