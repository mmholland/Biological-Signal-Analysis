clear; clc; close all;
Fs = 360; % sampling frequency -- I obtained this from the MIT-BIH database
nFs = Fs/2; % nyquist frequency
train = []; % array to store training data
test = []; % array to store test data
target = [0 ; 1 ; 2]; % target array
N = 2; % LPC order
k = 8; % sqrt of N is 1.4142, so I started with k = 1
psc = 0; % no. of psc heartbeats
pvc = 0; % no. of pvc heartbeats
normal = 0; % no. of normal heartbeats
files = 1500; % total no. of files
success = 0; % total no. of successful predictions

fid=fopen('train1500.txt'); %file that contains the names of data files
for i=1:files
    file=fgetl(fid); %read filename (one line at a time)
    load(file); %data variable to appear in workspace
      
    % remove the mean from each signal
    x_cond = x - mean(x);
    % convert values so value is between +1 and -1
    x_cond = x_cond/max(abs(x_cond));

    % detrend data by subtracting low order polynomial
    t = (1:length(x_cond))';
    opol = 6;
    [p,s,mu] = polyfit(t,x_cond,opol);
    f_y = polyval(p,t,[],mu);
    detrend_data = x_cond - f_y;
    
    % do filtering
    % 13th order lowpass Butterworth filter to remove high frequency noise
    % N = buttord((35/nFs), (40/nFs), 3, 30);
    [B, A] = butter(13, (35/nFs),'low');
    lowpassfilter = filtfilt(B, A, detrend_data);
    
    % 5th order highpass butterworth filter to remove baseline wander
    [B, A] = butter(5, (1/nFs),'high');
    result = filtfilt(B, A, lowpassfilter);
        
    % moving avg filter
    %L=10;
    %kernel=ones(L,1);
    %sm=filter(kernel,1,x_cond);

% do feature extraction
    featextraction = lpc(result,N);
    featextraction(:,1) = []; % remove first column

    % find R waves by detecting peaks
    % [~,locs_Rwave] = findpeaks(result,'MinPeakHeight',0.3,'MinPeakDistance',400);
    % plot(result);
    % plot(locs_Rwave,result(locs_Rwave),'rv','MarkerFaceColor','r');
    
    % find S waves by detecting peaks on an inverted signal
    % [~,locs_Swave] = findpeaks(-result(500:550));
    % [~,locs_Swave] = findpeaks(-result,'MinPeakHeight',0.1,'MinPeakDistance',400);
    % plot(locs_Swave,result(locs_Swave),'rs','MarkerFaceColor','b');
    
    % find Q wave
    % [~,min_locs] = findpeaks(-result,'MinPeakDistance',400,'MinPeakProminence',0.15);
    % locs_Qwave = min_locs(result(min_locs)>-0.135 & result(min_locs)<-0.08);
    % plot(locs_Qwave,result(locs_Qwave),'rs','MarkerFaceColor','g')

% save features from all data into an array
    train = [train;featextraction(1:end)];
end
fclose(fid);

fid=fopen('test1500.txt');
for i=1:files
file=fgetl(fid); %read filename (one line at a time)
load(file);
    
    % remove the mean from each signal
    x_cond = x - mean(x);
    % convert values so value is between +1 and -1
    x_cond = x_cond/max(abs(x_cond));

    % detrend data by subtracting low order polynomial
    t = (1:length(x_cond))';
    opol = 6;
    [p,s,mu] = polyfit(t,x_cond,opol);
    f_y = polyval(p,t,[],mu);
    detrend_data = x_cond - f_y;
    
% do filtering
    % 13th order lowpass Butterworth filter to remove high frequency noise
    [B, A] = butter(13, (35/nFs),'low');
    lowpassfilter = filtfilt(B, A, detrend_data);
    
    % 5th order highpass butterworth filter to remove baseline wander
    [B, A] = butter(5, (1/nFs),'high');
    result = filtfilt(B, A, lowpassfilter);

% do feature extraction
    featextraction = lpc(result,N); % perform lpc to Nth order
    featextraction(:,1) = []; % remove first column
    
% save features from all data into an array
    test = [test;featextraction(1:end)]; % store 
end
fclose(fid);

%do classification using train and test feature arrays
for i=1:499
    % create target array
    target = [target ; 0 ; 1 ; 2];
end
    
for i=1:files
    testing = test(i,:); % do one file at a time
    Edist = dist(train, testing'); % calculate Euclidean distance
    Et = [Edist, target]; % merge distance and target data
    sorted = sortrows(Et); % sort data
    predicted_class(i) = mode(sorted(1:k,2));

    % classifies signal into one of three groups
    if predicted_class(i) == 0
        normal = normal + 1;
    elseif predicted_class(i) == 1
        psc = psc + 1;
    else
        pvc = pvc + 1;
    end
    
    % increments if the classification was correct
    if predicted_class(i) == target(i)
        success = success + 1;
    end
    accuracy = (success/files)*100;
end